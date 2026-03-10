#!/usr/bin/env python3
"""
- phase=test：加载最新模型，对三种流型在test split上评估
- 数据：使用预处理目录 preprocessed_data
"""

import os
import json
import argparse
import time
import shutil
from typing import List, Dict, Tuple
from collections import deque
import random

import torch
# 启用TF32以提升A100等GPU上FP32 matmul性能
try:
	torch.set_float32_matmul_precision('high')
except Exception:
	pass
import numpy as np

from config import DEVICE, SCHEDULER_STEP_SIZE, SCHEDULER_GAMMA, USE_PREPROCESSED_DATA, PREPROCESSED_DIR, FLOWS, BATCH_SIZE, UNIFIED_TRAIN_EPOCHS, UNIFIED_TRAIN_LR, UNIFIED_OUT_DIR, GRADIENT_CLIP_NORM, QUICK_MODE, QUICK_MODE_RATIO, JOINT_TRAIN_EPOCHS, JOINT_TRAIN_LR, TORCH_COMPILE, AMP_MODE, ENABLE_INPUT_NORMALIZATION, SINGLE_TASK_EPOCHS, SINGLE_TASK_LR, ATTENTION_CONFIG, RANDOM_SEED, FLOW_PHYS_PARAMS
# 兼容从任意工作目录运行：将模型与数据处理目录加入 sys.path
import sys as _sys, os as _os
_ROOT_DIR = _os.path.dirname(_os.path.dirname(__file__))
_sys.path.insert(0, _os.path.join(_ROOT_DIR, 'models'))
_sys.path.insert(0, _os.path.join(_ROOT_DIR, 'data_processing'))
from network import Network
from pinn import PINN
from pdebench_h5_loader import load_split, iter_batches, FLOWS_DEFAULT, load_metadata


def parse_args(): # 命令行参数解析
	parser = argparse.ArgumentParser()
	parser.add_argument('--phase', type=str, default='train', choices=['train', 'test'])
	# 训练模式：joint（多任务联合）、single（单任务）
	parser.add_argument('--train-mode', type=str, default='joint', choices=['joint', 'single'])
	# 单任务指定流型
	parser.add_argument('--flow', type=str, default='', choices=FLOWS)
	# 单任务/联合训练保存
	parser.add_argument('--milestones', type=str, default='')
	# 允许命令行覆盖单任务总轮数（>0生效）
	parser.add_argument('--epochs', type=int, default=0)
	# 测试增强参数
	parser.add_argument('--exp-dir', type=str, default='')
	parser.add_argument('--viz', action='store_true')#测试时指定实验目录
	parser.add_argument('--viz-fields', type=str, default='u,v,p')#测试时指定模型权重
	parser.add_argument('--viz-time-idx', type=int, default=-1)#可视化
	# 新增：测试阶段可指定已有实验目录或模型权重
	parser.add_argument('--model-path', type=str, default='')
	# 其它参数从config读取，不强制命令行传参
	return parser.parse_args()


def make_experiment_dir(out_root: str) -> str: # 创建实验目录
	stamp = time.strftime('%Y%m%d_%H%M%S')
	exp_dir = os.path.join(out_root, stamp)
	os.makedirs(exp_dir, exist_ok=True)
	os.makedirs(os.path.join(exp_dir, 'models'), exist_ok=True)
	os.makedirs(os.path.join(exp_dir, 'metrics'), exist_ok=True)
	return exp_dir


def normalize_inputs(xyt: torch.Tensor, flow: str, pre_dir: str, pinn: PINN = None) -> torch.Tensor:
	"""根据流型的 metadata 与物理尺度对 (x, y, t) 进行归一化。
	
	- 空间：使用 metadata.domain 中的物理尺寸，将 x, y 线性缩放到 [0, 1]
	- 时间：使用特征时间尺度 T_char = L_char / U_char（来自 FLOW_PHYS_PARAMS）
	- 同时将尺度信息写入 PINN.norm_coeffs 以用于链式法则还原物理导数
	"""
	if not ENABLE_INPUT_NORMALIZATION:
		return xyt
	
	try:
		# 从 metadata 读取空间尺度
		meta = load_metadata(flow, pre_dir)
		domain = meta.get('domain', [1.0, 1.0])
		Lx_meta, Ly_meta = float(domain[0]), float(domain[1])
		
		# 从配置中读取特征长度 / 速度（若缺省则回退）
		phys = FLOW_PHYS_PARAMS.get(flow, {})
		L_char = float(phys.get('L', Lx_meta))
		U_char = float(phys.get('U', 1.0))
		
		# 空间归一化到 [0, 1]
		normalized = xyt.clone()
		normalized[:, 0] = xyt[:, 0] / Lx_meta  # x'
		normalized[:, 1] = xyt[:, 1] / Ly_meta  # y'
		
		# 特征时间尺度 T_char = L_char / U_char
		T_char = L_char / (U_char + 1e-8)
		normalized[:, 2] = xyt[:, 2] / T_char   # t'
		
		# 设置 PINN 的归一化系数，用于链式法则还原物理导数
		if pinn is not None:
			pinn.norm_coeffs[flow] = {
				'Lx': Lx_meta,
				'Ly': Ly_meta,
				'T':  T_char,
				'inv_Lx': 1.0 / Lx_meta,
				'inv_Ly': 1.0 / Ly_meta,
				'inv_T':  1.0 / T_char,
			}
		
		return normalized
	except Exception as e:
		print(f"Warning: Failed to normalize inputs for {flow}: {e}, using original")
		return xyt


def find_latest_exp_dir_unified(out_root: str) -> str:
	"""查找 results_unified/ 下最新的时间戳目录，返回绝对路径；不存在则返回空字符串。"""
	if not os.path.isdir(out_root):
		return ''
	subdirs = [d for d in os.listdir(out_root) if os.path.isdir(os.path.join(out_root, d))]
	if not subdirs:
		return ''
	subdirs.sort(reverse=True)
	return os.path.join(out_root, subdirs[0])


def compute_component_weights(prev_two: List[Dict[str, float]]) -> Dict[str, float]:
	"""基于最近两轮每个分量损失的均值，计算自适应权重（DWA风格）。"""
	components = ['eqn', 'div', 'psi', 'uv', 'data']
	# 不足两轮，等权
	if len(prev_two) < 2:
		return {c: 1.0 for c in components}
	last = prev_two[-1]
	prev = prev_two[-2]
	eps = 1e-8
	T = 2.0  # 温度
	raw = []
	for c in components:
		ratio = (last.get(c, 0.0) + eps) / (prev.get(c, 0.0) + eps)
		raw.append(np.exp(ratio / T))
	raw = np.array(raw, dtype=np.float64)
	# 归一化使权重总和为组件数，避免整体缩放影响学习率
	weights = raw / (raw.sum() + eps) * len(components)
	# 限制上下界，防止单项主导或被忽略
	weights = np.clip(weights, 0.2, 5.0)
	# 再次归一化到总和=组件数
	weights = weights / (weights.sum() + eps) * len(components)
	return {c: float(w) for c, w in zip(components, weights.tolist())}


def compute_flow_dwa_weights(
	prev_losses: Dict[str, List[float]],
	prev_weights: Dict[str, float],
	tau: float = 2.0,
	gamma: float = 0.9,
) -> Dict[str, float]:
	"""
	基于最近两轮每个流型总损失的相对改善率，计算跨流型的DWA权重 λ_i。
	实现方式与论文中的 DWA 思路一致：根据最近损失下降速度自适应调整各流型权重，
	并使用指数滑动平均进行平滑，避免权重剧烈震荡。
	"""
	flows = list(prev_losses.keys())
	eps = 1e-8
	# 若历史不足两轮，则所有流型等权
	if any(len(prev_losses[f]) < 2 for f in flows):
		return {f: 1.0 for f in flows}
	# 计算相对改善率 r_i^{(t)} = (L^{t-1} - L^{t}) / L^{t-1}
	rs: Dict[str, float] = {}
	for f in flows:
		hist = prev_losses[f]
		L_prev = float(hist[-2])
		L_last = float(hist[-1])
		rs[f] = (L_prev - L_last) / (abs(L_prev) + eps)
	# softmax 得到原始权重 λ_i
	raw_vals = np.array([np.exp(rs[f] / tau) for f in flows], dtype=np.float64)
	raw_sum = float(raw_vals.sum()) + eps
	lambdas = {f: float(raw_vals[i] / raw_sum) for i, f in enumerate(flows)}
	# 指数滑动平均平滑：\tilde{λ}_i^{(t)} = γ \tilde{λ}_i^{(t-1)} + (1-γ) λ_i^{(t)}
	new_weights: Dict[str, float] = {}
	for f in flows:
		prev_w = float(prev_weights.get(f, 1.0))
		new_weights[f] = gamma * prev_w + (1.0 - gamma) * lambdas[f]
	return new_weights


def train_phase(exp_dir: str, flows: List[str], pre_dir: str, batch_size: int, epochs: int, lr: float,
				stage_idx: int, network: Network, pinn: PINN,
				optimizer: torch.optim.Optimizer = None,
				scheduler: torch.optim.lr_scheduler._LRScheduler = None,
				start_epoch: int = 0) -> Tuple[Dict[str, float], Dict[str, Dict[str, List[float]]], torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
	model_dir = os.path.join(exp_dir, 'models')
	metrics_dir = os.path.join(exp_dir, 'metrics')

	# 允许外部传入已有优化器/调度器以实现连续训练
	if optimizer is None:
		# 64位精度使用更小的学习率
		optimizer = torch.optim.Adam(network.parameters(), lr=lr * 0.1)
	if scheduler is None:
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)

	loss_log: Dict[str, float] = {f: 0.0 for f in flows}
	# 记录每个流型最近两轮的分量损失均值，用于组件自适应权重
	prev_component_means: Dict[str, deque] = {f: deque(maxlen=2) for f in flows}
	# 记录每个流型最近两轮的总损失，用于跨流型DWA
	flow_total_losses: Dict[str, List[float]] = {f: [] for f in flows}
	flow_dwa_weights: Dict[str, float] = {f: 1.0 for f in flows}
	
	# 添加损失历史记录（用于画曲线）
	loss_history = {flow: {
		'weighted_total': [], 'unweighted_total': [], 
		'eqn': [], 'div': [], 'psi': [], 'uv': [], 'data': []
	} for flow in flows}
	
	for epoch in range(epochs):
		# 每个epoch累计各流型的分量损失与批次数
		epoch_sums: Dict[str, Dict[str, float]] = {f: {"weighted_total": 0.0, "unweighted_total": 0.0, "eqn": 0.0, "div": 0.0, "psi": 0.0, "uv": 0.0, "data": 0.0, "batches": 0} for f in flows}
		# 预先为每个流型计算本轮使用的组件权重
		flow_component_weights: Dict[str, Dict[str, float]] = {}
		for f in flows:
			w = compute_component_weights(list(prev_component_means[f]))
			# 早期课程：前15个epoch增强data，降低eqn/div，防止不稳定
			if epoch < 50:
				w['data'] *= 2.0
				w['eqn'] *= 0.8
				w['div'] *= 0.8
				# 归一化回到总和=5
				total_w = sum([w['eqn'], w['div'], w['psi'], w['uv'], w['data']]) + 1e-8
				for k in ['eqn', 'div', 'psi', 'uv', 'data']:
					w[k] = w[k] / total_w * 5.0
			# 未知边界模式：关闭硬边界损失（psi/uv），转而依赖方程残差（含边界方程项）
			try:
				from config import UNKNOWN_BC_MODE
			except Exception:
				UNKNOWN_BC_MODE = False
			if UNKNOWN_BC_MODE:
				w['psi'] = 0.0
				w['uv'] = 0.0
			flow_component_weights[f] = w
		for flow in flows:
			# 每流型：读取其metadata中的nu并设置到PINN（如果存在）
			try:
				meta = load_metadata(flow, pre_dir)
				if isinstance(meta, dict) and 'nu' in meta:
					try:
						pinn.nu = float(meta['nu'])
						# 可选：打印一次，便于核对
						print(f"[Train] Using nu={pinn.nu} for flow={flow}")
						
						if flow == 'pipe_flow' and abs(pinn.nu - 0.08) > 1e-9:
							raise RuntimeError(f"pipe_flow nu 应为 0.08，但读取为 {pinn.nu}。请检查 PREPROCESSED_DIR={pre_dir} 下的 {flow}/metadata.json 是否正确。")
					except Exception:
						pass
			except Exception:
				pass
			train_data = load_split(flow, 'train', pre_dir, quick_mode=QUICK_MODE, quick_ratio=QUICK_MODE_RATIO)
			for batch in iter_batches(train_data, batch_size=batch_size, shuffle=True):
				xyt_eqn = torch.tensor(batch['xyt_eqn'], dtype=torch.float64, device=DEVICE)
				xyt_bnd = torch.tensor(batch['xyt_bnd'], dtype=torch.float64, device=DEVICE)
				y_eqn = torch.tensor(batch['y_eqn'], dtype=torch.float64, device=DEVICE)
				y_div = torch.tensor(batch['y_div'], dtype=torch.float64, device=DEVICE)
				y_psi = torch.tensor(batch['y_psi_bnd'], dtype=torch.float64, device=DEVICE)
				y_uv = torch.tensor(batch['y_uv'], dtype=torch.float64, device=DEVICE)
				
				# 输入归一化
				xyt_eqn_norm = normalize_inputs(xyt_eqn, flow, pre_dir, pinn)
				xyt_bnd_norm = normalize_inputs(xyt_bnd, flow, pre_dir, pinn)

				# 获取真实标签（如果存在）
				y_psi_eqn = None
				y_p_eqn = None
				y_p_bnd = None
				if 'y_psi_eqn' in batch:
					y_psi_eqn = torch.tensor(batch['y_psi_eqn'], dtype=torch.float64, device=DEVICE)
				if 'y_p_eqn' in batch:
					y_p_eqn = torch.tensor(batch['y_p_eqn'], dtype=torch.float64, device=DEVICE)
				if 'y_p_bnd' in batch:
					y_p_bnd = torch.tensor(batch['y_p_bnd'], dtype=torch.float64, device=DEVICE)

				optimizer.zero_grad()
				model_flow = 'pipe' if flow == 'pipe_flow' else flow
				if AMP_MODE == 'bf16' and torch.cuda.is_available():
					with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
						weighted_total, unweighted_total, eqn, div, psi, uv, data, _ = pinn.compute_loss(
							xyt_eqn_norm, xyt_bnd_norm, y_eqn, y_div, y_psi, y_uv, model_flow,
							weights=flow_component_weights[flow],
							y_psi_eqn=y_psi_eqn, y_p_eqn=y_p_eqn, y_p_bnd=y_p_bnd
						)
				else:
					weighted_total, unweighted_total, eqn, div, psi, uv, data, _ = pinn.compute_loss(
						xyt_eqn_norm, xyt_bnd_norm, y_eqn, y_div, y_psi, y_uv, model_flow,
						weights=flow_component_weights[flow],
						y_psi_eqn=y_psi_eqn, y_p_eqn=y_p_eqn, y_p_bnd=y_p_bnd
					)
				# 检查损失是否为NaN
				if torch.isnan(weighted_total) or torch.isinf(weighted_total):
					print(f"Warning: NaN/Inf loss detected for {flow}, skipping this batch")
					continue
				
				# 跨流型DWA：对当前流型的总损失施加流型级权重
				scaled_loss = flow_dwa_weights.get(flow, 1.0) * weighted_total
				scaled_loss.backward()
				
				# 检查梯度是否为NaN
				has_nan_grad = False
				for param in network.parameters():
					if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
						has_nan_grad = True
						break
				
				if has_nan_grad:
					print(f"Warning: NaN/Inf gradients detected for {flow}, skipping this batch")
					optimizer.zero_grad()
					continue
				
				# 梯度裁剪防止梯度爆炸
				torch.nn.utils.clip_grad_norm_(network.parameters(), GRADIENT_CLIP_NORM)
				
				optimizer.step()

				# 统计分量损失
				epoch_sums[flow]["weighted_total"] += float(weighted_total.detach().cpu().item())
				epoch_sums[flow]["unweighted_total"] += float(unweighted_total.detach().cpu().item())
				epoch_sums[flow]["eqn"] += float(eqn.detach().cpu().item())
				epoch_sums[flow]["div"] += float(div.detach().cpu().item())
				epoch_sums[flow]["psi"] += float(psi.detach().cpu().item())
				epoch_sums[flow]["uv"] += float(uv.detach().cpu().item())
				epoch_sums[flow]["data"] += float(data.detach().cpu().item())
				epoch_sums[flow]["batches"] += 1

		# 学习率调度器：每个epoch只step一次
		scheduler.step()

		# 打印该epoch摘要
		current_lr = optimizer.param_groups[0]['lr']
		header = f"Stage {stage_idx+1} Epoch {start_epoch + epoch + 1}/{start_epoch + epochs}  lr={current_lr:.3e}"
		log_lines = [header]
		for flow in flows:
			b = max(1, epoch_sums[flow]["batches"])
			avg_weighted = epoch_sums[flow]["weighted_total"] / b
			avg_unweighted = epoch_sums[flow]["unweighted_total"] / b
			avg_eqn = epoch_sums[flow]["eqn"] / b
			avg_div = epoch_sums[flow]["div"] / b
			avg_psi = epoch_sums[flow]["psi"] / b
			avg_uv = epoch_sums[flow]["uv"] / b
			avg_data = epoch_sums[flow]["data"] / b
			w = flow_component_weights[flow]
			w_list = [w['eqn'], w['div'], w['psi'], w['uv'], w['data']]
			line1 = f"{flow}"
			line2 = f"| weighted={avg_weighted:.4e}   unweighted={avg_unweighted:.4e}"
			line3 = f"| eqn={avg_eqn:.4e}  div={avg_div:.4e}   psi={avg_psi:.4e}    uv={avg_uv:.4e}   data={avg_data:.4e}"
			line4 = f"| w={w_list}     batches={b}"
			log_lines.extend([line1, line2, line3, line4])
			
			# 记录损失历史（用于可视化）
			loss_history[flow]['weighted_total'].append(avg_weighted)
			loss_history[flow]['unweighted_total'].append(avg_unweighted)
			loss_history[flow]['eqn'].append(avg_eqn)
			loss_history[flow]['div'].append(avg_div)
			loss_history[flow]['psi'].append(avg_psi)
			loss_history[flow]['uv'].append(avg_uv)
			loss_history[flow]['data'].append(avg_data)
			# 保存本轮均值用于下一轮组件权重与流型级DWA计算
			prev_component_means[flow].append({
				'eqn': avg_eqn,
				'div': avg_div,
				'psi': avg_psi,
				'uv': avg_uv,
				'data': avg_data,
			})
			flow_total_losses[flow].append(avg_unweighted)
			
		# 基于当前epoch的各流型总损失，更新跨流型DWA权重
		flow_dwa_weights = compute_flow_dwa_weights(flow_total_losses, flow_dwa_weights)
		
		# 可选打印GPU显存
		if torch.cuda.is_available():
			mem_alloc = torch.cuda.memory_allocated() / (1024**3)
			mem_resv = torch.cuda.memory_reserved() / (1024**3)
			log_lines.append(f"  GPU mem: allocated={mem_alloc:.2f}GiB, reserved={mem_resv:.2f}GiB")
		print("\n".join(log_lines))
		print()

		# 维护阶段累计日志（用于阶段json）
		for flow in flows:
			loss_log[flow] += epoch_sums[flow]["weighted_total"]

	# 保存阶段模型（可续训快照）
	stage_path = os.path.join(model_dir, f'model_stage_{stage_idx+1}.pth')
	ckpt = {
		'network': network.state_dict(),
		'optimizer': optimizer.state_dict(),
		'scheduler': scheduler.state_dict(),
		'epoch': int(start_epoch + epochs),
		'mode': 'stage',
		'stage_idx': int(stage_idx + 1),
		'config': {
			'USE_PREPROCESSED_DATA': USE_PREPROCESSED_DATA,
			'PREPROCESSED_DIR': PREPROCESSED_DIR,
			'FLOWS': FLOWS,
			'AMP_MODE': AMP_MODE,
			'ENABLE_INPUT_NORMALIZATION': ENABLE_INPUT_NORMALIZATION,
		},
		'rng': {
			'torch': torch.get_rng_state(),
			'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
			'numpy': np.random.get_state(),
			'python': random.getstate(),
		}
	}
	torch.save(ckpt, stage_path)
	print(f"Saved stage {stage_idx+1} model to: {stage_path}")

	# 保存阶段损失摘要
	with open(os.path.join(metrics_dir, f'stage_{stage_idx+1}_summary.json'), 'w') as f:
		json.dump({f: loss_log[f] for f in flows}, f, indent=2)
	
	# 绘制损失曲线
	plot_loss_curves(loss_history, stage_idx+1, metrics_dir)
	
	return loss_log, loss_history, optimizer, scheduler


def test_phase(exp_dir: str, flows: List[str], pre_dir: str, batch_size: int, network: Network, pinn: PINN, model_path: str = None, args: argparse.Namespace = None):
	model_dir = os.path.join(exp_dir, 'models')
	# 使用最后阶段模型
	final_model_path = model_path if model_path else os.path.join(model_dir, 'model_stage_3.pth')
	if os.path.exists(final_model_path):
		state = torch.load(final_model_path, map_location=DEVICE)
		network.load_state_dict(state['network'])
		network.to(DEVICE)
		network.eval()
		print(f"Loaded model: {final_model_path}")
	else:
		print("Warning: final stage model not found; evaluating current weights.")

	metrics_dir = os.path.join(exp_dir, 'metrics')
	os.makedirs(metrics_dir, exist_ok=True)  # 确保metrics目录存在
	metrics: Dict[str, Dict[str, float]] = {}

	for flow in flows:
		# 每流型测试前：按metadata设置对应的nu（如果存在）
		try:
			meta = load_metadata(flow, pre_dir)
			if isinstance(meta, dict) and 'nu' in meta:
				try:
					pinn.nu = float(meta['nu'])
					print(f"[Test] Using nu={pinn.nu} for flow={flow}")
				except Exception:
					pass
		except Exception:
			pass
		test_data = load_split(flow, 'test', pre_dir)
		total_loss = 0.0
		count = 0
		# 新增：分量损失累计
		comp_sums = {"eqn": 0.0, "div": 0.0, "psi": 0.0, "uv": 0.0, "data": 0.0}
		for batch in iter_batches(test_data, batch_size=batch_size, shuffle=False):
			xyt_eqn = torch.tensor(batch['xyt_eqn'], dtype=torch.float64, device=DEVICE)
			xyt_bnd = torch.tensor(batch['xyt_bnd'], dtype=torch.float64, device=DEVICE)
			y_eqn = torch.tensor(batch['y_eqn'], dtype=torch.float64, device=DEVICE)
			y_div = torch.tensor(batch['y_div'], dtype=torch.float64, device=DEVICE)
			y_psi = torch.tensor(batch['y_psi_bnd'], dtype=torch.float64, device=DEVICE)
			y_uv = torch.tensor(batch['y_uv'], dtype=torch.float64, device=DEVICE)
			
			# 输入归一化
			xyt_eqn_norm = normalize_inputs(xyt_eqn, flow, pre_dir, pinn)
			xyt_bnd_norm = normalize_inputs(xyt_bnd, flow, pre_dir, pinn)
			
			model_flow = 'pipe' if flow == 'pipe_flow' else flow
			with torch.enable_grad():
				total, unweighted_total, eqn, div, psi, uv, data, _ = pinn.compute_loss(xyt_eqn_norm, xyt_bnd_norm, y_eqn, y_div, y_psi, y_uv, model_flow)
			total_loss += float(total.detach().cpu().item())
			# 新增：累计分量
			comp_sums["eqn"] += float(eqn.detach().cpu().item())
			comp_sums["div"] += float(div.detach().cpu().item())
			comp_sums["psi"] += float(psi.detach().cpu().item())
			comp_sums["uv"] += float(uv.detach().cpu().item())
			comp_sums["data"] += float(data.detach().cpu().item())
			count += 1
		avg_total = total_loss / max(1, count)
		# 新增：计算分量均值
		avg_components = {k: (v / max(1, count)) for k, v in comp_sums.items()}
		metrics[flow] = {"avg_total_loss": avg_total, **{f"avg_{k}": v for k, v in avg_components.items()}}
		print(f"Test {flow}: avg_total_loss={avg_total:.4e} over {count} batches | components: "
			  f"eqn={avg_components['eqn']:.3e}, div={avg_components['div']:.3e}, psi={avg_components['psi']:.3e}, uv={avg_components['uv']:.3e}, data={avg_components['data']:.3e}")

	with open(os.path.join(metrics_dir, 'test_metrics.json'), 'w') as f:
		json.dump(metrics, f, indent=2)
	print("Saved test metrics to:", os.path.join(metrics_dir, 'test_metrics.json'))

	# 可视化（全场GT存在时绘制u,v,p热力图）
	if args is not None and getattr(args, 'viz', False):
		for flow in flows:
			try:
				visualize_full_fields_unified(exp_dir, flow, network, args)
			except Exception as e:
				print(f"Warning: visualization for {flow} failed: {e}")


def plot_loss_curves(loss_history: Dict[str, Dict[str, List[float]]], stage_idx: int, metrics_dir: str, filename_suffix: str = ""):
	"""绘制损失曲线"""
	try:
		import matplotlib.pyplot as plt
		import matplotlib
		matplotlib.use('Agg')  # 使用非交互式后端
		
		# 创建子图
		fig, axes = plt.subplots(2, 2, figsize=(15, 10))
		fig.suptitle(f'Stage {stage_idx} Loss Curves', fontsize=16)
		
		# 颜色映射 - 为4个任务分配不同颜色
		color_map = {
			'lid_driven_cavity': 'blue',
			'pipe_flow': 'red', 
			'couette_flow': 'green',
			'shear_layer': 'orange'
		}
		flows = list(loss_history.keys())
		colors = [color_map.get(flow, 'purple') for flow in flows]  # 默认紫色
		
		# 1. 加权总损失 vs 未加权总损失
		ax1 = axes[0, 0]
		for i, flow in enumerate(flows):
			epochs = range(1, len(loss_history[flow]['weighted_total']) + 1)
			ax1.plot(epochs, loss_history[flow]['weighted_total'], 
					color=colors[i], label=f'{flow} (weighted)', linewidth=2)
			ax1.plot(epochs, loss_history[flow]['unweighted_total'], 
					color=colors[i], linestyle='--', label=f'{flow} (unweighted)', alpha=0.7)
		ax1.set_title('Total Loss Comparison')
		ax1.set_xlabel('Epoch')
		ax1.set_ylabel('Loss')
		ax1.legend()
		ax1.grid(True, alpha=0.3)
		ax1.set_yscale('log')
		
		# 2. 各分量损失
		ax2 = axes[0, 1]
		for i, flow in enumerate(flows):
			epochs = range(1, len(loss_history[flow]['eqn']) + 1)
			ax2.plot(epochs, loss_history[flow]['eqn'], 
					color=colors[i], label=f'{flow} (eqn)', linewidth=2)
		ax2.set_title('Equation Loss')
		ax2.set_xlabel('Epoch')
		ax2.set_ylabel('Loss')
		ax2.legend()
		ax2.grid(True, alpha=0.3)
		ax2.set_yscale('log')
		
		# 3. 边界条件损失
		ax3 = axes[1, 0]
		for i, flow in enumerate(flows):
			epochs = range(1, len(loss_history[flow]['psi']) + 1)
			ax3.plot(epochs, loss_history[flow]['psi'], 
					color=colors[i], label=f'{flow} (psi)', linewidth=2)
			ax3.plot(epochs, loss_history[flow]['uv'], 
					color=colors[i], linestyle='--', label=f'{flow} (uv)', alpha=0.7)
		ax3.set_title('Boundary Condition Loss')
		ax3.set_xlabel('Epoch')
		ax3.set_ylabel('Loss')
		ax3.legend()
		ax3.grid(True, alpha=0.3)
		ax3.set_yscale('log')
		
		# 4. 连续性方程损失
		ax4 = axes[1, 1]
		for i, flow in enumerate(flows):
			epochs = range(1, len(loss_history[flow]['div']) + 1)
			ax4.plot(epochs, loss_history[flow]['div'], 
					color=colors[i], label=f'{flow} (div)', linewidth=2)
		ax4.set_title('Divergence Loss')
		ax4.set_xlabel('Epoch')
		ax4.set_ylabel('Loss')
		ax4.legend()
		ax4.grid(True, alpha=0.3)
		ax4.set_yscale('log')
		
		plt.tight_layout()
		
		# 保存图片
		plot_path = os.path.join(metrics_dir, f'stage_{stage_idx}_loss_curves{filename_suffix}.png')
		plt.savefig(plot_path, dpi=300, bbox_inches='tight')
		plt.close()
		
		print(f"Saved loss curves to: {plot_path}")
		
		# 保存损失数据
		data_path = os.path.join(metrics_dir, f'stage_{stage_idx}_loss_data{filename_suffix}.json')
		with open(data_path, 'w') as f:
			json.dump(loss_history, f, indent=2)
		print(f"Saved loss data to: {data_path}")
		
	except ImportError:
		print("Warning: matplotlib not available, skipping loss curve plotting")
	except Exception as e:
		print(f"Warning: Failed to plot loss curves: {e}")


def visualize_full_fields_unified(exp_dir: str, flow: str, network: Network, args: argparse.Namespace) -> None:
	"""多任务模型在存在全场真值时，可视化指定流型与时刻的 u,v,p GT/Pred/Err 热力图。"""
	import numpy as np
	try:
		import matplotlib
		matplotlib.use('Agg')
		import matplotlib.pyplot as plt
	except Exception as e:
		print(f"matplotlib not available: {e}")
		return

	from config import PREPROCESSED_DIR
	meta_path = os.path.join(PREPROCESSED_DIR, flow, 'metadata.json')
	data_path = os.path.join(PREPROCESSED_DIR, flow, 'test_data.npz')
	if not os.path.exists(meta_path) or not os.path.exists(data_path):
		print(f"[{flow}] Full-field metadata or test_data.npz not found; skip visualization.")
		return
	meta = json.load(open(meta_path, 'r'))
	data_npz = np.load(data_path)

	def pick_key(cands: List[str]):
		for k in cands:
			if k in data_npz:
				return k
		return ''

	u_key = pick_key(['u_grid', 'u', 'U'])
	v_key = pick_key(['v_grid', 'v', 'V'])
	p_key = pick_key(['p_grid', 'p', 'P'])
	needed = {'u': u_key, 'v': v_key, 'p': p_key}
	missing = [k for k, v in needed.items() if v == '']
	if missing:
		print(f"[{flow}] Full-field ground truth missing for: {missing}; skip visualization.")
		return

	u_gt = data_npz[u_key]
	v_gt = data_npz[v_key]
	p_gt = data_npz[p_key]
	if u_gt.ndim != 3 or v_gt.ndim != 3 or p_gt.ndim != 3:
		print(f"[{flow}] GT arrays must be 3D (nt, ny, nx); skip visualization.")
		return
	nt, ny, nx = u_gt.shape
	time_idx = args.viz_time_idx if args.viz_time_idx >= 0 else (nt - 1)
	time_idx = max(0, min(nt - 1, time_idx))

	domain = meta.get('domain', [1, 1])
	gx, gy = meta.get('grid_size', [nx, ny])
	import numpy as np
	x = np.linspace(0.0, float(domain[0]), int(gx))
	y = np.linspace(0.0, float(domain[1]), int(gy))
	X, Y = np.meshgrid(x, y)
	t_list = meta.get('time_points', [0.0])
	t_val = float(t_list[time_idx]) if time_idx < len(t_list) else (t_list[-1] if len(t_list) > 0 else 0.0)
	xy = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1)
	t = np.full((xy.shape[0], 1), t_val, dtype=np.float32)
	xyt = np.concatenate([xy, t], axis=1).astype(np.float32)

	network.eval()
	tensor_xyt = torch.tensor(xyt, device=DEVICE, dtype=torch.float64, requires_grad=True)
	model_flow = 'pipe' if flow == 'pipe_flow' else flow
	out = network(tensor_xyt, model_flow)
	psi_pred = out[:, 0]
	p_pred = out[:, 1]
	grads = torch.autograd.grad(
		psi_pred, tensor_xyt, grad_outputs=torch.ones_like(psi_pred), retain_graph=False, create_graph=False
	)[0]
	dpsi_dx = grads[:, 0]
	dpsi_dy = grads[:, 1]
	u_pred = dpsi_dy
	v_pred = -dpsi_dx

	u_pred_np = u_pred.detach().cpu().numpy().reshape(ny, nx)
	v_pred_np = v_pred.detach().cpu().numpy().reshape(ny, nx)
	p_pred_np = p_pred.detach().cpu().numpy().reshape(ny, nx)

	u_gt_np = u_gt[time_idx]
	v_gt_np = v_gt[time_idx]
	p_gt_np = p_gt[time_idx]

	metrics_dir = os.path.join(exp_dir, 'metrics')
	fields = [f.strip() for f in args.viz_fields.split(',') if f.strip()]
	for name, gt, pred in [('u', u_gt_np, u_pred_np), ('v', v_gt_np, v_pred_np), ('p', p_gt_np, p_pred_np)]:
		if name not in fields:
			continue
		err = pred - gt
		vmin = float(np.min(gt))
		vmax = float(np.max(gt))
		err_abs = float(np.max(np.abs(err)))
		try:
			import matplotlib.pyplot as plt
			for kind, arr, cm, vmin_, vmax_ in [
				('gt', gt, 'viridis', vmin, vmax),
				('pred', pred, 'viridis', vmin, vmax),
				('err', err, 'bwr', -err_abs, err_abs),
			]:
				plt.figure(figsize=(5, 4))
				plt.imshow(arr, origin='lower', extent=[0, domain[0], 0, domain[1]], cmap=cm, vmin=vmin_, vmax=vmax_)
				plt.colorbar()
				plt.title(f"{flow} {name} {kind} t={time_idx}")
				out_path = os.path.join(metrics_dir, f"{flow}_{name}_{kind}_t{time_idx}.png")
				plt.tight_layout()
				plt.savefig(out_path, dpi=200)
				plt.close()
				print(f"Saved: {out_path}")
		except Exception as e:
			print(f"[{flow}] plotting failed: {e}")

	# 存数值
	import numpy as _np  # 防止命名污染
	_np.savez_compressed(
		os.path.join(metrics_dir, f"{flow}_viz_t{time_idx}.npz"),
		u_gt=u_gt_np, u_pred=u_pred_np, u_err=(u_pred_np - u_gt_np),
		v_gt=v_gt_np, v_pred=v_pred_np, v_err=(v_pred_np - v_gt_np),
		p_gt=p_gt_np, p_pred=p_pred_np, p_err=(p_pred_np - p_gt_np),
		X=X, Y=Y,
	)


def main():
	args = parse_args()
	flows = FLOWS

	if not USE_PREPROCESSED_DATA:
		raise RuntimeError('本入口仅支持使用预处理数据（请在config中将 USE_PREPROCESSED_DATA 设为 True）。')

	# 设置随机种子确保实验可重现性
	seed = RANDOM_SEED
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
	print(f"Random seed set to {seed} for reproducibility")

	# train 阶段创建新实验目录并写快照；test 阶段若提供 --exp-dir 则复用该目录
	if args.phase == 'train':
		exp_dir = make_experiment_dir(UNIFIED_OUT_DIR)
		with open(os.path.join(exp_dir, 'config_snapshot.json'), 'w') as f:
			json.dump({
				'USE_PREPROCESSED_DATA': USE_PREPROCESSED_DATA,
				'PREPROCESSED_DIR': PREPROCESSED_DIR,
				'FLOWS': FLOWS,
				'BATCH_SIZE': BATCH_SIZE,
				'UNIFIED_TRAIN_EPOCHS': UNIFIED_TRAIN_EPOCHS,
				'UNIFIED_TRAIN_LR': UNIFIED_TRAIN_LR,
				'TRAIN_MODE': args.train_mode,
				'JOINT_TRAIN_EPOCHS': JOINT_TRAIN_EPOCHS,
				'JOINT_TRAIN_LR': JOINT_TRAIN_LR,
				'SINGLE_TASK_EPOCHS': SINGLE_TASK_EPOCHS,
				'SINGLE_TASK_LR': SINGLE_TASK_LR
			}, f, indent=2)
	else:
		exp_dir = args.exp_dir if args.exp_dir else make_experiment_dir(UNIFIED_OUT_DIR)

	network = Network(ATTENTION_CONFIG).to(DEVICE).double()
	# 可选：torch.compile 加速
	if TORCH_COMPILE:
		try:
			network = torch.compile(network, mode='max-autotune')
		except Exception as e:
			print(f"Warning: torch.compile failed: {e}")
	pinn = PINN(network, rho=1.0, nu=0.1, u0=1.0)

	if args.phase == 'train':
		# 计时起点：总训练用时
		t0 = time.time()
		if args.train_mode == 'joint':
			# 自定义联合训练：在 2000、3000 epoch 处额外保存一次快照
			model_dir = os.path.join(exp_dir, 'models')
			os.makedirs(model_dir, exist_ok=True)
			# 复用 train_phase 的大部分逻辑：这里简单分段调用以确保精确保存点
			# 新增：维护累计损失历史（从第1轮到当前轮）
			cumulative_history = None
			# 第1段：训练到2000
			loss_log_1, loss_history_1, optimizer, scheduler = train_phase(
				exp_dir, FLOWS, PREPROCESSED_DIR, BATCH_SIZE, 2000, JOINT_TRAIN_LR, 2, network, pinn,
				optimizer=None, scheduler=None, start_epoch=0
			)
			# 更新累计历史并绘制“1~2000轮”损失
			cumulative_history = loss_history_1
			metrics_dir = os.path.join(exp_dir, 'metrics')
			plot_loss_curves(cumulative_history, 3, metrics_dir, filename_suffix='_epoch_2000')
			# 保存2000轮可续训快照
			ckpt_2000 = {
				'network': network.state_dict(),
				'optimizer': optimizer.state_dict(),
				'scheduler': scheduler.state_dict(),
				'epoch': 2000,
				'mode': 'joint_milestone',
				'config': {
					'USE_PREPROCESSED_DATA': USE_PREPROCESSED_DATA,
					'PREPROCESSED_DIR': PREPROCESSED_DIR,
					'FLOWS': FLOWS,
					'AMP_MODE': AMP_MODE,
					'ENABLE_INPUT_NORMALIZATION': ENABLE_INPUT_NORMALIZATION,
				},
				'rng': {
					'torch': torch.get_rng_state(),
					'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
					'numpy': np.random.get_state(),
					'python': random.getstate(),
				}
			}
			torch.save(ckpt_2000, os.path.join(model_dir, 'model_epoch_2000.pth'))
			print(f"Saved epoch 2000 model to: {os.path.join(model_dir, 'model_epoch_2000.pth')}")
			# 复制当下metrics（阶段内最新）到 epoch_2000 命名，便于追踪
			# 曲线图（若存在）与数据json
			for fname in [
				'stage_3_loss_curves.png', 'stage_2_loss_curves.png',
				'stage_3_loss_data.json', 'stage_2_loss_data.json',
				'stage_1_loss_data.json'
			]:
				src = os.path.join(metrics_dir, fname)
				if os.path.exists(src):
					base, ext = os.path.splitext(fname)
					shutil.copyfile(src, os.path.join(metrics_dir, f"{base}_epoch_2000{ext}"))
			# 汇总各流型阶段损失（若有 stage_3_summary.json）
			for sname in ['stage_3_summary.json', 'stage_2_summary.json', 'stage_1_summary.json']:
				src = os.path.join(metrics_dir, sname)
				if os.path.exists(src):
					shutil.copyfile(src, os.path.join(metrics_dir, sname.replace('.json', '_epoch_2000.json')))
			# 第2段：继续到3000（再训1000）
			loss_log_2, loss_history_2, optimizer, scheduler = train_phase(
				exp_dir, FLOWS, PREPROCESSED_DIR, BATCH_SIZE, 1000, JOINT_TRAIN_LR, 2, network, pinn,
				optimizer=optimizer, scheduler=scheduler, start_epoch=2000
			)
			# 追加到累计历史并绘制“1~3000轮”损失
			for flow in cumulative_history.keys():
				for k in cumulative_history[flow].keys():
					cumulative_history[flow][k].extend(loss_history_2[flow][k])
			plot_loss_curves(cumulative_history, 3, metrics_dir, filename_suffix='_epoch_3000')
			# 保存3000轮可续训快照
			ckpt_3000 = {
				'network': network.state_dict(),
				'optimizer': optimizer.state_dict(),
				'scheduler': scheduler.state_dict(),
				'epoch': 3000,
				'mode': 'joint_milestone',
				'config': {
					'USE_PREPROCESSED_DATA': USE_PREPROCESSED_DATA,
					'PREPROCESSED_DIR': PREPROCESSED_DIR,
					'FLOWS': FLOWS,
					'AMP_MODE': AMP_MODE,
					'ENABLE_INPUT_NORMALIZATION': ENABLE_INPUT_NORMALIZATION,
				},
				'rng': {
					'torch': torch.get_rng_state(),
					'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
					'numpy': np.random.get_state(),
					'python': random.getstate(),
				}
			}
			torch.save(ckpt_3000, os.path.join(model_dir, 'model_epoch_3000.pth'))
			print(f"Saved epoch 3000 model to: {os.path.join(model_dir, 'model_epoch_3000.pth')}")
			# 复制当下metrics到 epoch_3000 命名
			for fname in [
				'stage_3_loss_curves.png', 'stage_2_loss_curves.png',
				'stage_3_loss_data.json', 'stage_2_loss_data.json',
				'stage_1_loss_data.json'
			]:
				src = os.path.join(metrics_dir, fname)
				if os.path.exists(src):
					base, ext = os.path.splitext(fname)
					shutil.copyfile(src, os.path.join(metrics_dir, f"{base}_epoch_3000{ext}"))
			for sname in ['stage_3_summary.json', 'stage_2_summary.json', 'stage_1_summary.json']:
				src = os.path.join(metrics_dir, sname)
				if os.path.exists(src):
					shutil.copyfile(src, os.path.join(metrics_dir, sname.replace('.json', '_epoch_3000.json')))
			# 第3段：继续到5000（再训2000）
			remaining = max(0, JOINT_TRAIN_EPOCHS - 3000)
			if remaining > 0:
				loss_log_3, loss_history_3, optimizer, scheduler = train_phase(
					exp_dir, FLOWS, PREPROCESSED_DIR, BATCH_SIZE, remaining, JOINT_TRAIN_LR, 2, network, pinn,
					optimizer=optimizer, scheduler=scheduler, start_epoch=3000
				)
				# 追加到累计历史并绘制“1~5000轮”损失
				for flow in cumulative_history.keys():
					for k in cumulative_history[flow].keys():
						cumulative_history[flow][k].extend(loss_history_3[flow][k])
				plot_loss_curves(cumulative_history, 3, metrics_dir, filename_suffix=f"_epoch_{3000+remaining}")
				# 保存5000轮可续训快照
				ckpt_last = {
					'network': network.state_dict(),
					'optimizer': optimizer.state_dict(),
					'scheduler': scheduler.state_dict(),
					'epoch': int(3000 + remaining),
					'mode': 'joint_milestone',
					'config': {
						'USE_PREPROCESSED_DATA': USE_PREPROCESSED_DATA,
						'PREPROCESSED_DIR': PREPROCESSED_DIR,
						'FLOWS': FLOWS,
						'AMP_MODE': AMP_MODE,
						'ENABLE_INPUT_NORMALIZATION': ENABLE_INPUT_NORMALIZATION,
					},
					'rng': {
						'torch': torch.get_rng_state(),
						'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
						'numpy': np.random.get_state(),
						'python': random.getstate(),
					}
				}
				torch.save(ckpt_last, os.path.join(model_dir, f'model_epoch_{3000+remaining}.pth'))
			print('Joint training finished. You can run test phase to evaluate.')
			# 统计并保存训练时间
			elapsed = time.time() - t0
			epochs_run = JOINT_TRAIN_EPOCHS
			avg_sec_per_epoch = (elapsed / max(1, epochs_run))
			training_time = {
				'total_seconds': float(elapsed),
				'epochs_run': int(epochs_run),
				'avg_seconds_per_epoch': float(avg_sec_per_epoch)
			}
			with open(os.path.join(exp_dir, 'metrics', 'training_time.json'), 'w') as f:
				json.dump(training_time, f, indent=2)
			print(f"[Timing] total={elapsed:.2f}s, epochs={epochs_run}, avg/epoch={avg_sec_per_epoch:.3f}s")
		elif args.train_mode == 'single':
			if args.flow == '':
				raise ValueError('单任务模式需要指定 --flow，取值为: ' + ','.join(FLOWS))
			# 使用联合训练第三阶段的 epochs 与 lr 作为默认单任务配置，或按需调整
			single_epochs = SINGLE_TASK_EPOCHS
			single_lr = SINGLE_TASK_LR
			# 若命令行提供 --epochs 且 >0，则覆盖默认单任务轮数
			if hasattr(args, 'epochs') and isinstance(args.epochs, int) and args.epochs > 0:
				single_epochs = int(args.epochs)
			# 解析里程碑（如有），确保升序且不超过总轮数
			milestones: List[int] = []
			if args.milestones.strip():
				try:
					milestones = sorted({int(x) for x in args.milestones.split(',') if x.strip().isdigit() and int(x) > 0 and int(x) <= single_epochs})
				except Exception:
					milestones = []
			# 按里程碑分段训练并保存可续训快照；无里程碑则一次性训练
			optimizer = None
			scheduler = None
			last_epoch = 0
			segments = milestones + ([single_epochs] if (not milestones or milestones[-1] < single_epochs) else [])
			for m in segments:
				seg_epochs = m - last_epoch
				_, _, optimizer, scheduler = train_phase(
					exp_dir, [args.flow], PREPROCESSED_DIR, BATCH_SIZE, seg_epochs, single_lr, 0, network, pinn,
					optimizer=optimizer, scheduler=scheduler, start_epoch=last_epoch
				)
				# 保存当前里程碑的可续训快照
				model_dir = os.path.join(exp_dir, 'models')
				os.makedirs(model_dir, exist_ok=True)
				ckpt = {
					'network': network.state_dict(),
					'optimizer': optimizer.state_dict() if optimizer is not None else {},
					'scheduler': scheduler.state_dict() if scheduler is not None else {},
					'epoch': int(m),
					'mode': 'single_milestone',
					'flow': args.flow,
					'config': {
						'USE_PREPROCESSED_DATA': USE_PREPROCESSED_DATA,
						'PREPROCESSED_DIR': PREPROCESSED_DIR,
						'FLOWS': [args.flow],
						'AMP_MODE': AMP_MODE,
						'ENABLE_INPUT_NORMALIZATION': ENABLE_INPUT_NORMALIZATION,
					},
					'rng': {
						'torch': torch.get_rng_state(),
						'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
						'numpy': np.random.get_state(),
						'python': random.getstate(),
					}
				}
				torch.save(ckpt, os.path.join(model_dir, f'model_epoch_{m}.pth'))
				print(f"Saved single-task milestone checkpoint to: {os.path.join(model_dir, f'model_epoch_{m}.pth')}")
				last_epoch = m
			print(f'Single-task training for {args.flow} finished. You can run test phase to evaluate.')
		else:
			raise ValueError('Unknown train mode')
	elif args.phase == 'test':
		model_path = args.model_path if args.model_path else None
		test_phase(exp_dir, FLOWS, PREPROCESSED_DIR, BATCH_SIZE, network, pinn, model_path=model_path, args=args)
	else:
		raise ValueError('Unknown phase')


if __name__ == '__main__':
	main() 