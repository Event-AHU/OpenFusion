import torch
import torch.nn as nn
from torch.autograd import grad
from typing import Dict, Tuple
import contextlib
from network import Network

class PINN(nn.Module):
    """物理信息神经网络模型"""
    
    def __init__(self, network: Network, rho: float = 1.0, nu: float = 0.1, u0: float = 1.0):
        """初始化PINN模型
        
        Args:
            network: 神经网络模型
            rho: 流体密度
            nu: 运动粘度
            u0: 特征速度
        """
        super().__init__()
        self.network = network
        self.device = next(network.parameters()).device
        self.rho = rho
        self.nu = nu
        self.u0 = u0
        # 归一化系数缓存
        self.norm_coeffs = {}
    
    def get_normalization_coeffs(self, xyt: torch.Tensor, flow_type: str) -> dict:
        """获取归一化系数，用于链式法则还原物理导数
        
        Args:
            xyt: 输入坐标张量
            flow_type: 流型类型
            
        Returns:
            包含归一化系数的字典
        """
        # 检查输入是否已经归一化（通过检查坐标范围）
        x_range = xyt[:, 0].max() - xyt[:, 0].min()
        y_range = xyt[:, 1].max() - xyt[:, 1].min()
        t_range = xyt[:, 2].max() - xyt[:, 2].min()
        
        # 如果坐标范围在 [0,1] 附近，说明已经归一化
        if (x_range < 1.5 and y_range < 1.5 and t_range < 1.5):
            # 已归一化，需要还原系数
            if flow_type not in self.norm_coeffs:
                # 默认系数（如果metadata不可用）
                self.norm_coeffs[flow_type] = {
                    'Lx': 1.0, 'Ly': 1.0, 'T': 1.0,
                    'inv_Lx': 1.0, 'inv_Ly': 1.0, 'inv_T': 1.0
                }
            return self.norm_coeffs[flow_type]
        else:
            # 未归一化，系数为1
            return {
                'Lx': 1.0, 'Ly': 1.0, 'T': 1.0,
                'inv_Lx': 1.0, 'inv_Ly': 1.0, 'inv_T': 1.0
            }
    
    def compute_gradients(self, xyt: torch.Tensor, flow_type: str) -> Dict[str, torch.Tensor]:
        """计算所有需要的梯度，优化版本"""
        # 在AMP环境（bf16/fp16）下禁用autocast，避免高阶梯度不可用
        autocast_off = torch.autocast(device_type='cuda', enabled=False) if torch.cuda.is_available() else contextlib.nullcontext()
        with autocast_off:
            xyt = xyt.clone().requires_grad_(True)
            # 计算流函数和压力
            psi_p = self.network(xyt, flow_type)
        psi = psi_p[:, 0:1]
        p = psi_p[:, 1:2]
        
        # 获取归一化系数，用于链式法则还原物理导数
        norm_coeffs = self.get_normalization_coeffs(xyt, flow_type)
        inv_Lx, inv_Ly, inv_T = norm_coeffs['inv_Lx'], norm_coeffs['inv_Ly'], norm_coeffs['inv_T']
        
        # 分别计算 ψ 与 p 对 x,y 的一阶导数
        psi_grad = torch.autograd.grad(
            psi,
            xyt,
            grad_outputs=torch.ones_like(psi),
            create_graph=True,
            retain_graph=True,
            allow_unused=False
        )[0]
        p_grad = torch.autograd.grad(
            p,
            xyt,
            grad_outputs=torch.ones_like(p),
            create_graph=True,
            retain_graph=True,
            allow_unused=False
        )[0]
        
        # 基于 ψ 的梯度计算速度分量；基于 p 的梯度得到压力梯度
        # 应用链式法则：∂/∂x = (1/Lx) * ∂/∂x', ∂/∂y = (1/Ly) * ∂/∂y'
        dpsi_dx = psi_grad[:, 0:1] * inv_Lx
        dpsi_dy = psi_grad[:, 1:2] * inv_Ly
        u = dpsi_dy
        v = -dpsi_dx
        p_x = p_grad[:, 0:1] * inv_Lx
        p_y = p_grad[:, 1:2] * inv_Ly
        
        # 计算速度对时间的导数
        u_t = torch.autograd.grad(
            u,
            xyt,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
            allow_unused=False
        )[0][:, 2:3] * inv_T  # 应用链式法则：∂/∂t = (1/T) * ∂/∂t'
        v_t = torch.autograd.grad(
            v,
            xyt,
            grad_outputs=torch.ones_like(v),
            create_graph=True,
            retain_graph=True,
            allow_unused=False
        )[0][:, 2:3] * inv_T  # 应用链式法则：∂/∂t = (1/T) * ∂/∂t'
        
        # 计算速度对空间的一阶导数
        u_grads = torch.autograd.grad(
            u,
            xyt,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
            allow_unused=False
        )[0]
        v_grads = torch.autograd.grad(
            v,
            xyt,
            grad_outputs=torch.ones_like(v),
            create_graph=True,
            retain_graph=True,
            allow_unused=False
        )[0]
        
        # 应用链式法则：∂/∂x = (1/Lx) * ∂/∂x', ∂/∂y = (1/Ly) * ∂/∂y'
        u_x = u_grads[:, 0:1] * inv_Lx
        u_y = u_grads[:, 1:2] * inv_Ly
        v_x = v_grads[:, 0:1] * inv_Lx
        v_y = v_grads[:, 1:2] * inv_Ly
        
        # 计算速度对空间的二阶导数
        u_xx = torch.autograd.grad(
            u_x,
            xyt,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
            retain_graph=True,
            allow_unused=False
        )[0][:, 0:1] * inv_Lx  # 应用链式法则：∂²/∂x² = (1/Lx²) * ∂²/∂x'²
        u_yy = torch.autograd.grad(
            u_y,
            xyt,
            grad_outputs=torch.ones_like(u_y),
            create_graph=True,
            retain_graph=True,
            allow_unused=False
        )[0][:, 1:2] * inv_Ly  # 应用链式法则：∂²/∂y² = (1/Ly²) * ∂²/∂y'²
        v_xx = torch.autograd.grad(
            v_x,
            xyt,
            grad_outputs=torch.ones_like(v_x),
            create_graph=True,
            retain_graph=True,
            allow_unused=False
        )[0][:, 0:1] * inv_Lx  # 应用链式法则：∂²/∂x² = (1/Lx²) * ∂²/∂x'²
        v_yy = torch.autograd.grad(
            v_y,
            xyt,
            grad_outputs=torch.ones_like(v_y),
            create_graph=True,
            retain_graph=True,
            allow_unused=False
        )[0][:, 1:2] * inv_Ly  # 应用链式法则：∂²/∂y² = (1/Ly²) * ∂²/∂y'²
        
        return {
            'u': u, 'v': v,
            'p_x': p_x, 'p_y': p_y,
            'u_t': u_t, 'v_t': v_t,
            'u_x': u_x, 'u_y': u_y,
            'v_x': v_x, 'v_y': v_y,
            'u_xx': u_xx, 'u_yy': u_yy,
            'v_xx': v_xx, 'v_yy': v_yy
        }
    
    def compute_loss(self, xyt_eqn: torch.Tensor, xyt_bnd: torch.Tensor, 
                    y_eqn: torch.Tensor, y_div: torch.Tensor, y_psi: torch.Tensor, y_uv: torch.Tensor,
                    flow_type: str, weights: Dict[str, float] = None,
                    y_psi_eqn: torch.Tensor = None, y_p_eqn: torch.Tensor = None,
                    y_p_bnd: torch.Tensor = None) -> Tuple[torch.Tensor, ...]:
        """计算PINN损失
        
        Args:
            xyt_eqn: 方程点坐标 (batch_size, 3)
            xyt_bnd: 边界点坐标 (batch_size, 3)
            y_eqn: 方程残差目标值 (batch_size, 2)
            y_div: 散度目标值 (batch_size, 1)
            y_psi: 流函数边界条件目标值 (batch_size, 1)
            y_uv: 速度边界条件目标值 (batch_size, 2)
            flow_type: 流动类型
            weights: 损失权重字典
            y_psi_eqn: 方程点真实流函数值 (batch_size, 1) - 可选
            y_p_eqn: 方程点真实压力值 (batch_size, 1) - 可选
            y_p_bnd: 边界点真实压力值 (batch_size, 1) - 可选
            
        Returns:
            Tuple[torch.Tensor, ...]: (weighted_total, unweighted_total, eqn_loss, div_loss, psi_loss, uv_loss, data_loss, grads_dict)
        """
        if weights is None:
            weights = {'eqn': 1.0, 'div': 1.0, 'psi': 1.0, 'uv': 1.0, 'data': 1.0}
        
        # 分别计算内部点和边界点的梯度，减少内存使用
        eqn_grads = self.compute_gradients(xyt_eqn, flow_type)
        bnd_grads = self.compute_gradients(xyt_bnd, flow_type)
        
        # 动量方程残差
        f_x = (eqn_grads['u_t'] + eqn_grads['u'] * eqn_grads['u_x'] + 
               eqn_grads['v'] * eqn_grads['u_y'] + eqn_grads['p_x'] / self.rho - 
               self.nu * (eqn_grads['u_xx'] + eqn_grads['u_yy']))
        
        f_y = (eqn_grads['v_t'] + eqn_grads['u'] * eqn_grads['v_x'] + 
               eqn_grads['v'] * eqn_grads['v_y'] + eqn_grads['p_y'] / self.rho - 
               self.nu * (eqn_grads['v_xx'] + eqn_grads['v_yy']))
        
        eqn_internal = torch.mean((torch.cat([f_x, f_y], dim=1) - y_eqn) ** 2)

        # 新增：将边界点的方程残差并入 eqn 损失（目标为0）
        # 目的：在未知边界条件时仍对边界处解进行物理约束
        try:
            from config import BOUNDARY_EQN_WEIGHT
        except Exception:
            BOUNDARY_EQN_WEIGHT = 0.0
        bnd_eqn_loss = torch.tensor(0.0, device=self.device)
        if xyt_bnd is not None and xyt_bnd.numel() > 0 and BOUNDARY_EQN_WEIGHT > 0.0:
            # 复用已计算的 bnd_grads
            f_x_b = (bnd_grads['u_t'] + bnd_grads['u'] * bnd_grads['u_x'] +
                     bnd_grads['v'] * bnd_grads['u_y'] + bnd_grads['p_x'] / self.rho -
                     self.nu * (bnd_grads['u_xx'] + bnd_grads['u_yy']))
            f_y_b = (bnd_grads['v_t'] + bnd_grads['u'] * bnd_grads['v_x'] +
                     bnd_grads['v'] * bnd_grads['v_y'] + bnd_grads['p_y'] / self.rho -
                     self.nu * (bnd_grads['v_xx'] + bnd_grads['v_yy']))
            bnd_eqn_loss = torch.mean(torch.cat([f_x_b, f_y_b], dim=1) ** 2)
        eqn_loss = eqn_internal + (BOUNDARY_EQN_WEIGHT * bnd_eqn_loss)
        
        # 连续性方程残差
        div_loss = torch.mean((eqn_grads['u_x'] + eqn_grads['v_y'] - y_div) ** 2)
        
        # 边界条件损失
        psi_bnd = self.network(xyt_bnd, flow_type)[:, 0:1]
        u_bnd = bnd_grads['u']
        v_bnd = bnd_grads['v']
        
        psi_loss = torch.mean((psi_bnd - y_psi) ** 2)
        uv_loss = torch.mean((torch.cat([u_bnd, v_bnd], dim=1) - y_uv) ** 2)
        
        # 数据拟合损失（如果有真实标签）
        data_loss = torch.tensor(0.0, device=self.device)
        if y_psi_eqn is not None or y_p_eqn is not None or y_p_bnd is not None:
            data_losses = []
            
            # 方程点数据拟合
            if y_psi_eqn is not None:
                psi_eqn = self.network(xyt_eqn, flow_type)[:, 0:1]
                psi_data_loss = torch.mean((psi_eqn - y_psi_eqn) ** 2)
                data_losses.append(psi_data_loss)
            
            if y_p_eqn is not None:
                p_eqn = self.network(xyt_eqn, flow_type)[:, 1:2]
                p_data_loss = torch.mean((p_eqn - y_p_eqn) ** 2)
                data_losses.append(p_data_loss)
            
            # 边界点数据拟合
            if y_p_bnd is not None:
                p_bnd = self.network(xyt_bnd, flow_type)[:, 1:2]
                p_bnd_data_loss = torch.mean((p_bnd - y_p_bnd) ** 2)
                data_losses.append(p_bnd_data_loss)
            
            if data_losses:
                data_loss = torch.mean(torch.stack(data_losses))
        
        # 计算各损失项的梯度大小
        loss_dict = {
            'eqn': eqn_loss,
            'div': div_loss,
            'psi': psi_loss,
            'uv': uv_loss,
            'data': data_loss
        }
        
        # 使用torch.no_grad()计算梯度范数，避免创建计算图
        grads_dict = {}
        with torch.no_grad():
            for key, loss in loss_dict.items():
                try:
                    grad_tensors = torch.autograd.grad(loss, self.network.parameters(), 
                                                     retain_graph=True, create_graph=False,
                                                     allow_unused=True)
                    grad_norm = sum(g.norm().item() for g in grad_tensors if g is not None)
                    grads_dict[key] = grad_norm
                except RuntimeError:
                    # 如果某个损失项没有梯度，设为0
                    grads_dict[key] = 0.0
        
        # 计算未加权总损失
        unweighted_total = sum(loss_dict.values())
        
        # 加权总损失：若传入weights则使用之，否则各项等权
        if weights is not None:
            w_eqn = float(weights.get('eqn', 1.0))
            w_div = float(weights.get('div', 1.0))
            w_psi = float(weights.get('psi', 1.0))
            w_uv = float(weights.get('uv', 1.0))
            w_data = float(weights.get('data', 1.0))
        else:
            w_eqn = w_div = w_psi = w_uv = w_data = 1.0
        
        weighted_total = (
            w_eqn * eqn_loss +
            w_div * div_loss +
            w_psi * psi_loss +
            w_uv * uv_loss +
            w_data * data_loss
        )
        
        # 返回加权和未加权的损失
        return weighted_total, unweighted_total, eqn_loss, div_loss, psi_loss, uv_loss, data_loss, grads_dict
    
    def forward_uv(self, xyt: torch.Tensor, flow_type: str = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """预测速度场
        
        Args:
            xyt: 输入张量
            flow_type: 流动类型
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (u, v)
        """
        grads = self.compute_gradients(xyt, flow_type)
        return grads['u'], grads['v']
    
    def predict(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, 
               flow_type: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """预测流场
        
        Args:
            x: x坐标张量
            y: y坐标张量
            t: 时间张量
            flow_type: 流动类型
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (u, v, p)
        """
        self.eval()
        with torch.no_grad():
            # 准备输入
            xyt = torch.stack([x.flatten(), y.flatten(), t.flatten()], dim=1)
            
            # 计算流函数和压力
            psi_p = self.network(xyt, flow_type)
            psi = psi_p[:, 0:1]
            p = psi_p[:, 1:2]
            
            # 计算速度场
            xyt.requires_grad_(True)
            with torch.enable_grad():
                psi_grad = grad(psi, xyt, grad_outputs=torch.ones_like(psi), 
                              create_graph=False)[0]
            u = psi_grad[:, 1:2]  # du/dy
            v = -psi_grad[:, 0:1]  # -du/dx
            
            # 重塑输出
            u = u.reshape_as(x)
            v = v.reshape_as(y)
            p = p.reshape_as(x)
            
            return u, v, p 