import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import random
import matplotlib.pyplot as plt
import argparse

try:
    from smoothing_spline import Smoothing_spline
except ImportError:
    print("Warning: smoothing_spline module not found or cvxopt missing. SmoothingSpline will not be available.")
    Smoothing_spline = None



DATA_DIR = os.path.join('/wangx/DATA/Dataset/Ti_Te_new_data/new_data_11')
INTERP_POINTS = 32  
TARGET_GRID = np.linspace(0, 1, INTERP_POINTS)
SEED = 42


EXCLUDE_COLS = {'Ti_out', 'Te_out', 'Shotnum', 't', 'timestamp', 'Spec'}


def set_seed(seed):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")

def parse_array_string(s, reshape_to_pairs=False, keep_error=False):
    
    if not isinstance(s, str):
        return np.array([])
    
    
    s_clean = s.replace('[', ' ').replace(']', ' ').strip()
    if not s_clean:
        return np.array([])
    
    try:
        
        values = np.fromstring(s_clean, sep=' ')
        
        if reshape_to_pairs:
            
            row_splits = s.split(']')
            if len(row_splits) > 1:
                
                first_row = row_splits[0].replace('[', ' ').strip()
                cols = len(first_row.split())
            else:
                
                cols = 3 if values.size % 3 == 0 and values.size % 2 != 0 else 2

            if cols == 3 and values.size % 3 == 0:
                
                data = values.reshape(-1, 3)
                if keep_error:
                    return data 
                else:
                    return data[:, :2] 
            elif cols == 2 and values.size % 2 == 0:
                
                data = values.reshape(-1, 2)
                if keep_error:
                    
                    err_col = np.zeros((data.shape[0], 1))
                    return np.hstack([data, err_col])
                else:
                    return data
            else:
                 
                 if values.size % 2 == 0:
                     data = values.reshape(-1, 2)
                     if keep_error:
                         err_col = np.zeros((data.shape[0], 1))
                         return np.hstack([data, err_col])
                     return data
                 print(f"Warning: Data size {values.size} not divisible by detected columns {cols}.")
                 if keep_error:
                     return np.zeros((0, 3))
                 else:
                     return np.zeros((0, 2))
        else:
            return values
            
    except Exception as e:
        print(f"Warning: Failed to parse array string: {e}")
        if reshape_to_pairs:
            if keep_error:
                return np.zeros((0, 3))
            else:
                return np.zeros((0, 2))
        else:
            return np.array([])


def process_ti_profile(ti_data, te_processed, target_grid):
    
    try:
       
        mask_ti = (
            ~np.isnan(ti_data[:, 1]) & 
            (ti_data[:, 1] > 0) & 
            ((ti_data[:, 2] / ti_data[:, 1]) <= 0.5)
        )
        
        ti_rho_valid = ti_data[mask_ti, 0]
        ti_val_valid = ti_data[mask_ti, 1]
        
        if len(ti_rho_valid) < 2:
            
            return np.zeros_like(target_grid)

        
        sort_idx = np.argsort(ti_rho_valid)
        ti_rho_valid = ti_rho_valid[sort_idx]
        ti_val_valid = ti_val_valid[sort_idx]
        
        
        rho_min_ti = ti_rho_valid[0]
        rho_max_ti = ti_rho_valid[-1]
        
        
        f_ti = interp1d(ti_rho_valid, ti_val_valid, kind='linear', bounds_error=False, fill_value=np.nan)
        
        
        ti_interp_vals = f_ti(target_grid)
        
        
        valid_indices = np.where(~np.isnan(ti_interp_vals))[0]
        
        if len(valid_indices) == 0:
            return np.zeros_like(target_grid)
            
        last_idx = valid_indices[-1]
        
        
        if last_idx == len(target_grid) - 1:
            combined_vals = ti_interp_vals
        else:
            
            ti_last_val = ti_interp_vals[last_idx]
            
            
            te_join_val = te_processed[last_idx]
            
            
            if te_join_val > 1.0: 
                factor = ti_last_val / te_join_val
            else:
                factor = 0.0 
                
          
            te_tail = te_processed[last_idx+1:]
            
           
            ti_tail_scaled = te_tail * factor
            
            
            combined_vals = np.concatenate([ti_interp_vals[:last_idx+1], ti_tail_scaled])
            
       
        mask_nan = np.isnan(combined_vals)
        if mask_nan.any():
            
            valid_mask = ~mask_nan
            if valid_mask.any():
                first_valid = combined_vals[valid_mask][0]
                combined_vals[mask_nan] = first_valid
            else:
                return np.zeros_like(target_grid)

        
        if Smoothing_spline is not None:
            try:
               
                spline = Smoothing_spline(x=target_grid, y=combined_vals, w=np.ones_like(target_grid), lamda=0.0021)
                spline.fit()
                final_val = spline.eval(target_grid)
            except Exception as e:
                
                final_val = combined_vals 
        else:
            final_val = combined_vals
            
        
        final_val[final_val < 0] = 0.0
        
        
        diffs = np.diff(final_val)
        
        
        if len(final_val) > 0:
            
            max_idx = np.argmax(final_val)
            
            
            if target_grid[max_idx] >= 0.1:
                return None  
                
            
            max_increase = np.max(diffs) if len(diffs) > 0 else 0
            if max_increase > max_val * 0.15 or max_increase > 100.0: 
                return None
                
        return final_val
        
    except Exception as e:
        print(f"Error in process_ti_profile: {e}")
        return None

def masked_mse_loss(output, target, dataset):
    
    target_orig = dataset.get_original_y(target)
    mask = (target_orig > 0.0).float() # 0.0 for true value > 0
    
   
    diff = output - target
    squared_err = diff ** 2
    
    
    masked_err = squared_err * mask
    
    
    valid_count = mask.sum()
    
    if valid_count > 0:
        mse_loss = masked_err.sum() / valid_count
        
        
        ti_out = output[:, :32]
        te_out = output[:, 32:]
        
        
        ti_diff = ti_out[:, 1:] - ti_out[:, :-1]
        te_diff = te_out[:, 1:] - te_out[:, :-1]
        
        
        mono_penalty = torch.relu(ti_diff).mean() + torch.relu(te_diff).mean()
        
        
        loss = mse_loss + 0.1 * mono_penalty
        
    else:
        loss = torch.tensor(0.0, device=output.device, requires_grad=True)
        
    return loss, valid_count, masked_err.sum()

def evaluate_model(model, dataloader, criterion, dataset, device='cpu'):
    
    model.eval()
    total_mse_sum = 0.0
    total_mae_sum = 0.0
    total_valid_samples = 0
    
    
    total_target_sum = 0.0
    total_target_sq = 0.0
    
    with torch.no_grad():
        for macro_inputs, spec_inputs, targets, _ in dataloader:
            macro_inputs = macro_inputs.to(device)
            spec_inputs = spec_inputs.to(device)
            targets = targets.to(device)
            
            outputs, _ = model(macro_inputs, spec_inputs)
            
            
            _, count, batch_mse_sum = masked_mse_loss(outputs, targets, dataset)
            
    
            target_orig = dataset.get_original_y(targets)
            mask = (target_orig > 0.0).float()
            abs_err = torch.abs(outputs - targets) * mask
            batch_mae_sum = abs_err.sum()
            

            total_mse_sum += batch_mse_sum.item()
            total_mae_sum += batch_mae_sum.item()
            total_valid_samples += count.item()
            
   
            masked_targets = targets * mask
            total_target_sum += masked_targets.sum().item()
            total_target_sq += (masked_targets ** 2).sum().item()
            
    if total_valid_samples > 0:
        avg_mse = total_mse_sum / total_valid_samples
        avg_mae = total_mae_sum / total_valid_samples
        rmse = np.sqrt(avg_mse)
        
 
        target_mean = total_target_sum / total_valid_samples
  
        target_var = (total_target_sq / total_valid_samples) - (target_mean ** 2)
        
        if target_var > 1e-8:
            r2 = 1 - (avg_mse / target_var)
        else:
            r2 = 0.0
    else:
        avg_mse, avg_mae, rmse, r2 = 0.0, 0.0, 0.0, 0.0
        
    return avg_mse, avg_mae, rmse, r2


   
def visualize_interpolation(dataset, save_name='interpolation_plot.png'):
 
    if not hasattr(dataset, 'interp_viz_data') or not dataset.interp_viz_data:
        print("No interpolation data available for visualization.")
        return

    viz_data = dataset.interp_viz_data
    num_samples = len(viz_data)
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 4 * num_samples))
    
    if num_samples == 1:
        axes = [axes] 
        
    for i, data in enumerate(viz_data):
        shotnum = data['shotnum']
        grid = data['grid']
        
        ti_raw = data['ti_raw']
        ti_interp = data['ti_interp']
        te_raw = data['te_raw']
        te_interp = data['te_interp']
        
        
        ax_ti = axes[i][0]
        if ti_raw.size > 0:
            
            rho_raw = ti_raw[:, 0]
            val_raw = ti_raw[:, 1]
            
            
            if ti_raw.shape[1] > 2:
                err_raw = ti_raw[:, 2]
                
                max_display_err = 2000.0
                err_display = np.clip(err_raw, 0, max_display_err)
                ax_ti.errorbar(rho_raw, val_raw, yerr=err_display, fmt='none', ecolor='blue', alpha=0.3, zorder=4)
            
            ax_ti.scatter(rho_raw, val_raw, color='blue', label='Raw Ti', alpha=0.6, zorder=5)
                
        ax_ti.plot(grid, ti_interp, color='red', label='Interpolated Ti', linewidth=2)
        ax_ti.set_title(f'Shot {shotnum} - Ti Profile')
        ax_ti.set_xlabel('rho')
        ax_ti.set_ylabel('Ti (eV)')
        ax_ti.grid(True, alpha=0.3)
        
        
        max_ti_val = np.max(ti_interp) if len(ti_interp) > 0 else 5000
        if ti_raw.size > 0:
            valid_raw_max = np.percentile(ti_raw[:, 1], 95) 
            max_ti_val = max(max_ti_val, valid_raw_max)
        ax_ti.set_ylim(bottom=0, top=max_ti_val * 1.2) 
        
        ax_ti.legend()
        
        
        ax_te = axes[i][1]
        if te_raw.size > 0:
            rho_raw = te_raw[:, 0]
            val_raw = te_raw[:, 1]
            ax_te.scatter(rho_raw, val_raw, color='blue', label='Raw Te', alpha=0.6, zorder=5)
            
        ax_te.plot(grid, te_interp, color='red', label='Interpolated Te', linewidth=2)
        ax_te.set_title(f'Shot {shotnum} - Te Profile')
        ax_te.set_xlabel('rho')
        ax_te.set_ylabel('Te (eV)')
        ax_te.grid(True, alpha=0.3)
        ax_te.legend()
        
    plt.tight_layout()
    plt.savefig(save_name)
    plt.close(fig)
    print(f"Interpolation Visualization saved to {save_name}")

def visualize_results(model, dataloader, dataset, save_name='prediction_plot.png', device='cpu'):
    
    
    model.eval()
    
    
    target_samples = 16
    collected_macro_inputs = []
    collected_spec_inputs = []
    collected_targets = []
    collected_shots = []
    
    
    with torch.no_grad():
        for macro_inputs, spec_inputs, targets, shots in dataloader:
            collected_macro_inputs.append(macro_inputs)
            collected_spec_inputs.append(spec_inputs)
            collected_targets.append(targets)
            collected_shots.append(shots)
            if sum(len(x) for x in collected_macro_inputs) >= target_samples:
                break
                
    if not collected_macro_inputs:
        print("No data to visualize.")
        return

    macro_inputs = torch.cat(collected_macro_inputs)[:target_samples].to(device)
    spec_inputs = torch.cat(collected_spec_inputs)[:target_samples].to(device)
    targets = torch.cat(collected_targets)[:target_samples]
    shots_all = torch.cat(collected_shots)[:target_samples]
    
    
    with torch.no_grad():
        outputs, _ = model(macro_inputs, spec_inputs)
        outputs = outputs.cpu()
    
    
    targets_denorm = dataset.get_original_y(targets)
    outputs_denorm = dataset.get_original_y(outputs)
    
    
    num_samples = macro_inputs.size(0)
    cols = 4
    rows = (num_samples + cols - 1) // cols
    
    grid_rho = np.linspace(0, 1, 32)
    
    
    fig_ti, axes_ti = plt.subplots(rows, cols, figsize=(20, 5*rows))
    axes_ti = axes_ti.flatten() if num_samples > 1 else [axes_ti]
    
    for i in range(num_samples):
        true_profile = targets_denorm[i].numpy()
        pred_profile = outputs_denorm[i].numpy()
        shot_num = int(shots_all[i].item())
        
        if np.all(true_profile == 0):
            print(f"Visualization Sample {i+1} (Shot {shot_num}): Ground Truth is all zeros.")

        
        true_ti = true_profile[:32]
        pred_ti = pred_profile[:32]
        
        
        mse = np.mean((true_ti - pred_ti)**2)
        
        ax = axes_ti[i]
        ax.plot(grid_rho, true_ti, 'b-', label='True')
        ax.plot(grid_rho, pred_ti, 'r--', label='Pred')
        ax.set_title(f'Shot {shot_num} (MSE: {mse:.1f})')
        ax.set_xlabel('rho')
        ax.set_ylabel('Ti (eV)')
        if i == 0: ax.legend() 
        ax.grid(True, alpha=0.3)
        
    
    for j in range(num_samples, len(axes_ti)):
        axes_ti[j].axis('off')
        
    plt.tight_layout()
    save_name_ti = save_name.replace('.png', '_Ti_soomth.png')
    plt.savefig(save_name_ti)
    plt.close(fig_ti)
    print(f"Ti Visualization saved to {save_name_ti}")

    
    fig_te, axes_te = plt.subplots(rows, cols, figsize=(20, 5*rows))
    axes_te = axes_te.flatten() if num_samples > 1 else [axes_te]
    
    for i in range(num_samples):
        true_profile = targets_denorm[i].numpy()
        pred_profile = outputs_denorm[i].numpy()
        shot_num = int(shots_all[i].item())
        
        
        true_te = true_profile[32:]
        pred_te = pred_profile[32:]
        
        mse = np.mean((true_te - pred_te)**2)
        
        ax = axes_te[i]
        ax.plot(grid_rho, true_te, 'b-', label='True')
        ax.plot(grid_rho, pred_te, 'r--', label='Pred')
        ax.set_title(f'Shot {shot_num} (MSE: {mse:.1f})')
        ax.set_xlabel('rho')
        ax.set_ylabel('Te (eV)')
        if i == 0: ax.legend()
        ax.grid(True, alpha=0.3)
        
    
    for j in range(num_samples, len(axes_te)):
        axes_te[j].axis('off')
        
    plt.tight_layout()
    save_name_te = save_name.replace('.png', '_Te_soomth.png')
    plt.savefig(save_name_te)
    plt.close(fig_te)
    print(f"Te Visualization saved to {save_name_te}")

    
    epsilon = 1e-6
    
    
    ti_true_all = targets_denorm[:num_samples, :32].numpy()
    ti_pred_all = outputs_denorm[:num_samples, :32].numpy()
    ti_rel_error = (ti_true_all - ti_pred_all) / (ti_true_all + epsilon)
    
    
    te_true_all = targets_denorm[:num_samples, 32:].numpy()
    te_pred_all = outputs_denorm[:num_samples, 32:].numpy()
    te_rel_error = (te_true_all - te_pred_all) / (te_true_all + epsilon)

    
    fig_err, axes_err = plt.subplots(2, 2, figsize=(15, 12))
    
    
    for i in range(num_samples):
        axes_err[0, 0].plot(grid_rho, ti_rel_error[i], alpha=0.5, linewidth=1)
    axes_err[0, 0].axhline(0, color='k', linestyle='--', linewidth=1)
    axes_err[0, 0].set_title('Ti Relative Error Profiles ((True-Pred)/True)')
    axes_err[0, 0].set_xlabel('rho')
    axes_err[0, 0].set_ylabel('Relative Error')
    axes_err[0, 0].set_ylim(-1.0, 1.0) 
    axes_err[0, 0].grid(True, alpha=0.3)

    
    for i in range(num_samples):
        axes_err[0, 1].plot(grid_rho, te_rel_error[i], alpha=0.5, linewidth=1)
    axes_err[0, 1].axhline(0, color='k', linestyle='--', linewidth=1)
    axes_err[0, 1].set_title('Te Relative Error Profiles ((True-Pred)/True)')
    axes_err[0, 1].set_xlabel('rho')
    axes_err[0, 1].set_ylabel('Relative Error')
    axes_err[0, 1].set_ylim(-1.0, 1.0)
    axes_err[0, 1].grid(True, alpha=0.3)

    
    axes_err[1, 0].hist(ti_rel_error.flatten(), bins=50, range=(-1, 1), alpha=0.7, color='blue', edgecolor='black')
    axes_err[1, 0].set_title('Ti Relative Error Distribution')
    axes_err[1, 0].set_xlabel('Relative Error')
    axes_err[1, 0].set_ylabel('Count')
    axes_err[1, 0].grid(True, alpha=0.3)

    
    axes_err[1, 1].hist(te_rel_error.flatten(), bins=50, range=(-1, 1), alpha=0.7, color='red', edgecolor='black')
    axes_err[1, 1].set_title('Te Relative Error Distribution')
    axes_err[1, 1].set_xlabel('Relative Error')
    axes_err[1, 1].set_ylabel('Count')
    axes_err[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_name_err = save_name.replace('.png', '_RelError.png')
    plt.savefig(save_name_err)
    plt.close(fig_err)
    print(f"Relative Error Visualization saved to {save_name_err}")


def visualize_attention_weights(model, dataloader, dataset, save_name='feature_importance.png', device='cpu'):
    
    model.eval()
    all_weights = []
    
    with torch.no_grad():
        for macro_inputs, spec_inputs, _, _ in dataloader:
            macro_inputs = macro_inputs.to(device)
            spec_inputs = spec_inputs.to(device)
            
            _, weights = model(macro_inputs, spec_inputs)
            all_weights.append(weights.cpu())
            
    if not all_weights:
        print("No weights to visualize.")
        return
        
    all_weights = torch.cat(all_weights, dim=0) 
    avg_weights = all_weights.mean(dim=0).numpy() 
    feature_names = dataset.feature_names
    
   
    sorted_indices = np.argsort(avg_weights)[::-1] 
    sorted_weights = avg_weights[sorted_indices]
    sorted_names = [feature_names[i] for i in sorted_indices]
    
    plt.figure(figsize=(12, 8))
   
    plt.barh(range(len(sorted_names)), sorted_weights[::-1], color='skyblue', edgecolor='black')
    plt.yticks(range(len(sorted_names)), sorted_names[::-1])
    plt.xlabel('Average Attention Weight (0-1)')
    plt.title('Macro Feature Importance (SE Attention Weights)')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(save_name)
    plt.close()
    print(f"Feature Importance Visualization saved to {save_name}")



class TokamakDataset(Dataset):
    def __init__(self, data_dir):
        self.file_paths = glob.glob(os.path.join(data_dir, '*.csv'))
        if not self.file_paths:
            print(f"Warning: No CSV files found in {data_dir}")
        else:
            print(f"Found {len(self.file_paths)} data files.")
            
        
        self.feature_names = []
        if self.file_paths:
            self._inspect_features(self.file_paths[0])
            
        
        self.X_macro = []
        self.X_spec = []
        self.Y_data = []
        self.shotnums = []
        
        self.interp_viz_data = [] 
        self._load_all_data()
        
        
        self.X_macro = torch.stack(self.X_macro)
        self.X_spec = torch.stack(self.X_spec)
        self.Y_data = torch.stack(self.Y_data)
        
        print(f"Dataset Loaded. Macro shape: {self.X_macro.shape}, Spec shape: {self.X_spec.shape}, Y shape: {self.Y_data.shape}")
        
        
        self.x_macro_mean = torch.zeros(self.X_macro.shape[1])
        self.x_macro_std = torch.ones(self.X_macro.shape[1])
        
        
        self.x_spec_mean = torch.zeros(1)
        self.x_spec_std = torch.ones(1)
        
        self.y_mean = torch.zeros(self.Y_data.shape[1])
        self.y_std = torch.ones(self.Y_data.shape[1])
        self.normalize = False

    def _inspect_features(self, file_path):
        try:
            df = pd.read_csv(file_path)
            cols = [c for c in df.columns if c not in EXCLUDE_COLS]
            self.feature_names = cols
            
            self.input_dim = len(cols)
            print(f"Input Features ({self.input_dim}): {cols}")
        except Exception as e:
            print(f"Error inspecting features: {e}")

    def _load_all_data(self):
        print("Loading data into memory...")
        for file_path in self.file_paths:
            try:
                
                df = pd.read_csv(file_path)
                
                
                for row_idx in range(len(df)):
                    
                    try:
                        
                        def get_scalar_val(col_name):
                            if col_name not in df.columns:
                                return 0.0
                            val_raw = df[col_name].iloc[row_idx]
                            if isinstance(val_raw, str):
                                
                                parsed = parse_array_string(val_raw, reshape_to_pairs=False)
                                return parsed[0] if parsed.size > 0 else 0.0
                            return float(val_raw)

                        
                        ne1 = get_scalar_val('ne1_val')
                        ne2 = get_scalar_val('ne2_val')
                        
                        if abs(ne1 - ne2) > 1.0:
                            
                            continue

                        
                        te_raw_str = df['Te_out'].iloc[row_idx]
                        te_raw_arr = parse_array_string(te_raw_str, reshape_to_pairs=True)

                       
                        if te_raw_arr.shape[0] > 20:
                            
                            continue

                        
                        wp_val = get_scalar_val('Wp_kJ')
                        
                        if te_raw_arr.size > 0:
                            max_te = np.max(te_raw_arr[:, 1]) 
                            
                           
                            if max_te > 8000.0 and wp_val < 150.0:
                                
                                continue
                                
                            
                            if max_te < 100.0:
                                
                                continue

                        
                        if 'Spec' not in df.columns:
                            
                            continue
                            
                        spec_raw_str = df['Spec'].iloc[row_idx]
                        if not isinstance(spec_raw_str, str):
                            
                            continue
                            
                        
                        shot_val = get_scalar_val('Shotnum')
                        shot_num_current = int(shot_val)
                        
                        
                        spec_arr = parse_array_string(spec_raw_str, reshape_to_pairs=False)
                        
                        
                        if spec_arr.size != 1891 * 487:
                            
                            continue
                            
                    
                        spec_matrix = spec_arr.reshape((1891, 487))
                        
                

                        if shot_num_current <= 102880:
                            
                            spec_slice_487 = spec_matrix[:, 290:356]
                        elif shot_num_current < 124768:
                            
                            spec_slice_487 = spec_matrix[:, 237:303]
                        else:
                            
                            spec_slice_487 = spec_matrix[:, 260:326]
                            
                        
                        spec_intervals = [
                            (213, 251), (252, 290), (291, 329), (330, 368), (369, 407),
                            (425, 463), (464, 502), (503, 541), (542, 580), (581, 619),
                            (637, 675), (676, 714), (715, 753), (754, 792), (793, 831),
                            (849, 887), (888, 926), (927, 965), (966, 1004), (1005, 1043),
                            (1061, 1099), (1100, 1138), (1139, 1177), (1178, 1216), (1217, 1255),
                            (1273, 1311), (1312, 1350), (1351, 1389), (1390, 1428), (1429, 1467),
                            (1485, 1523), (1524, 1562), (1563, 1601), (1602, 1640), (1641, 1679)
                        ]
                        
                        spec_processed = []
                        for start_idx, end_idx in spec_intervals:
                           
                            interval_sum = np.sum(spec_slice_487[start_idx:end_idx+1, :], axis=0)
                            spec_processed.append(interval_sum)
                            
                        
                        spec_features_2d = np.stack(spec_processed) 

                    except Exception as e:
                        print(f"Warning: Error filtering row {row_idx} in {file_path}: {e}")
                        continue
                    
                    x_values = []
                    for col in self.feature_names:
                        val_str = df[col].iloc[row_idx]
                        if isinstance(val_str, str):
                            try:
                                parsed = parse_array_string(val_str, reshape_to_pairs=False)
                                val = parsed[0] if parsed.size > 0 else 0.0
                            except:
                                val = 0.0
                        else:
                            val = float(val_str)
                        
                       
                        if col.endswith('_MW') and val < 0.01:
                            val = 0.0
                            
                        x_values.append(val)
                        
                   
                    spec_val = spec_features_2d
                    
                    
                    ti_str = df['Ti_out'].iloc[row_idx]
                    te_str = df['Te_out'].iloc[row_idx]
                    
                    
                    ti_arr = parse_array_string(ti_str, reshape_to_pairs=True, keep_error=True)
                    te_arr = parse_array_string(te_str, reshape_to_pairs=True, keep_error=False)
                    
                    
                    if ti_arr.size > 0:
                        ti_arr[:, 1] = ti_arr[:, 1] * 1000.0
                        
                        if ti_arr.shape[1] > 2:
                            ti_arr[:, 2] = ti_arr[:, 2] * 1000.0
                            
                    
                    if te_arr.size > 0:
                        
                        sort_idx = np.argsort(te_arr[:, 0])
                        te_sorted = te_arr[sort_idx]
                        
                        
                        first_val = te_sorted[0, 1]
                        
                        
                        synthetic_val = first_val * 1.01
                        
                        if te_arr.shape[1] == 3: 
                            new_point = np.array([[0.0, synthetic_val, 0.0]])
                        else:
                            new_point = np.array([[0.0, synthetic_val]])
                            
                        
                        te_arr = np.vstack([new_point, te_arr])
                    
                    te_interp = interpolate_profile(te_arr, TARGET_GRID)
                    
                    
                    core_mask = TARGET_GRID <= 0.2
                    if np.any(core_mask):
                        te_core = te_interp[core_mask]
                        te_core_diff = np.max(te_core) - np.min(te_core)
                        if te_core_diff < 50.0:
                            
                            continue
                    
                    ti_interp = process_ti_profile(ti_arr, te_interp, TARGET_GRID)
                    
                    
                    def is_strictly_monotonic(profile, tolerance=10.0):
                        
                        diff = profile[1:] - profile[:-1]
                        return not np.any(diff > tolerance)

                    if not is_strictly_monotonic(ti_interp, tolerance=10.0):
                        
                        continue
                        
                    if not is_strictly_monotonic(te_interp, tolerance=10.0):
                        
                        continue
                    
                    y_val = np.concatenate([ti_interp, te_interp])
                    
                    y_val[y_val < 0.0] = 0.0
                    y_val[y_val > 10000.0] = 0.0
                    
                    
                    if np.all(y_val[:32] == 0) or np.all(y_val[32:] == 0):
                        continue
                    
                   
                    if 'Shotnum' in df.columns:
                        shot_val = df['Shotnum'].iloc[row_idx]
                        if isinstance(shot_val, str):
                            try:
                                parsed_shot = parse_array_string(shot_val)
                                shot_num = int(parsed_shot[0]) if parsed_shot.size > 0 else 0
                            except:
                                shot_num = 0
                        else:
                            shot_num = int(shot_val)
                    else:
                        shot_num = 0
                    
                    self.shotnums.append(shot_num)
                    self.X_macro.append(torch.tensor(x_values, dtype=torch.float32))
                    
                    spec_tensor = torch.tensor(spec_val, dtype=torch.float32).unsqueeze(0)
                    self.X_spec.append(spec_tensor)
                    self.Y_data.append(torch.tensor(y_val, dtype=torch.float32))
                    
                   
                    if len(self.interp_viz_data) < 5:
                        self.interp_viz_data.append({
                            'shotnum': shot_num,
                            'ti_raw': ti_arr,
                            'ti_interp': ti_interp,
                            'te_raw': te_arr,
                            'te_interp': te_interp,
                            'grid': TARGET_GRID
                        })
                    
                    if te_arr.shape[1] == 3: 
                        new_point = np.array([[0.0, synthetic_val, 0.0]])
                    else:
                        new_point = np.array([[0.0, synthetic_val]])
                        
                   
                    te_arr = np.vstack([new_point, te_arr])
                
                te_interp = interpolate_profile(te_arr, TARGET_GRID)
                
                
                core_mask = TARGET_GRID <= 0.2
                if np.any(core_mask):
                    te_core = te_interp[core_mask]
                    te_core_diff = np.max(te_core) - np.min(te_core)
                    if te_core_diff < 50.0:
                        
                        continue
               
                ti_interp = process_ti_profile(ti_arr, te_interp, TARGET_GRID)
                
                
                if ti_interp is None:
                    
                    continue
                
                
                def is_strictly_monotonic(profile, tolerance=10.0):
                    
                    diff = profile[1:] - profile[:-1]
                    return not np.any(diff > tolerance)

                if not is_strictly_monotonic(ti_interp, tolerance=10.0):
                    
                    continue
                    
                if not is_strictly_monotonic(te_interp, tolerance=10.0):
                    
                    continue
               
                y_val = np.concatenate([ti_interp, te_interp])
                
               
                
                y_val[y_val < 0.0] = 0.0
                y_val[y_val > 10000.0] = 0.0
                
                
                if np.all(y_val[:32] == 0) or np.all(y_val[32:] == 0):
                    
                    continue
                
                
                if 'Shotnum' in df.columns:
                    shot_val = df['Shotnum'].iloc[0]
                    if isinstance(shot_val, str):
                        try:
                            parsed_shot = parse_array_string(shot_val)
                            shot_num = int(parsed_shot[0]) if parsed_shot.size > 0 else 0
                        except:
                            shot_num = 0
                    else:
                        shot_num = int(shot_val)
                else:
                    shot_num = 0
                
                self.shotnums.append(shot_num)
                self.X_macro.append(torch.tensor(x_values, dtype=torch.float32))
                
                spec_tensor = torch.tensor(spec_val, dtype=torch.float32).unsqueeze(0)
                self.X_spec.append(spec_tensor)
                self.Y_data.append(torch.tensor(y_val, dtype=torch.float32))
                
                
                if len(self.interp_viz_data) < 5:
                    self.interp_viz_data.append({
                        'shotnum': shot_num,
                        'ti_raw': ti_arr,
                        'ti_interp': ti_interp,
                        'te_raw': te_arr,
                        'te_interp': te_interp,
                        'grid': TARGET_GRID
                    })
                
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

    def set_normalization(self, x_macro_mean, x_macro_std, x_spec_mean, x_spec_std, y_mean, y_std):
        self.x_macro_mean = x_macro_mean
        self.x_macro_std = x_macro_std
        self.x_macro_std[self.x_macro_std == 0] = 1.0
        
        self.x_spec_mean = x_spec_mean
        self.x_spec_std = x_spec_std
        self.x_spec_std[self.x_spec_std == 0] = 1.0
        
        self.y_mean = y_mean
        self.y_std = y_std
        self.y_std[self.y_std == 0] = 1.0
        
        self.normalize = True
        print("Normalization enabled.")

    def __len__(self):
        return len(self.X_macro)

    def __getitem__(self, idx):
        x_m = self.X_macro[idx]
        x_s = self.X_spec[idx]
        y = self.Y_data[idx]
        shot = self.shotnums[idx]
        
        if self.normalize:
            x_m = (x_m - self.x_macro_mean) / self.x_macro_std
            x_s = (x_s - self.x_spec_mean) / self.x_spec_std
            y = (y - self.y_mean) / self.y_std
            
        return x_m, x_s, y, shot

    def get_original_y(self, y_normalized):
        """Helper to denormalize Y (for interpretation)"""
        if not self.normalize:
            return y_normalized
        y_std = self.y_std.to(y_normalized.device)
        y_mean = self.y_mean.to(y_normalized.device)
        return y_normalized * y_std + y_mean


class MacroFeatureAttention(nn.Module):
    
    def __init__(self, input_dim):
        super(MacroFeatureAttention, self).__init__()
        hidden_dim = max(8, input_dim // 2) 
        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1), 
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid() 
        )
        
    def forward(self, x):
        
        weights = self.attention_net(x)
        
        attended_x = x + (x * weights)
        return attended_x, weights

class Spec2DEncoder(nn.Module):
   
    def __init__(self, output_dim=128):
        super(Spec2DEncoder, self).__init__()
        self.conv_net = nn.Sequential(
            
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2), 
            
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2) 
        )
        
        self.fc = nn.Sequential(
            
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Dropout(0.4),
            
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, output_dim)
        )
        
    def forward(self, x):
        features = self.conv_net(x)
        features = features.view(features.size(0), -1) 
        return self.fc(features)

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob=0.4):
        super(SimpleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class DualMLP(nn.Module):
    
    def __init__(self, input_dim, output_dim, dropout_prob=0.4):
        super(DualMLP, self).__init__()
        
        
        assert output_dim % 2 == 0, 
        self.single_profile_dim = output_dim // 2
        
        
        self.ti_net = SimpleMLP(input_dim, self.single_profile_dim, dropout_prob)
        
        
        self.te_net = SimpleMLP(input_dim, self.single_profile_dim, dropout_prob)
        
    def forward(self, x):
        ti_out = self.ti_net(x)
        te_out = self.te_net(x)
        
        
        return torch.cat([ti_out, te_out], dim=1)
    
class DualResMLP(nn.Module):
   
    def __init__(self, input_dim, output_dim, hidden_dim=256, num_blocks=3, dropout_prob=0.2):
        super(DualResMLP, self).__init__()
        
        
        assert output_dim % 2 == 0, "Output dim must be divisible by 2 for DualResMLP"
        self.single_profile_dim = output_dim // 2
        
        
        self.ti_net = ResMLP(input_dim, self.single_profile_dim, hidden_dim, num_blocks, dropout_prob)
        
        
        self.te_net = ResMLP(input_dim, self.single_profile_dim, hidden_dim, num_blocks, dropout_prob)
        
    def forward(self, x):
        ti_out = self.ti_net(x)
        te_out = self.te_net(x)
        
        
        return torch.cat([ti_out, te_out], dim=1)

class PhysicsGuidedLoss(nn.Module):
   
    def __init__(self, dataset, core_weight=2.0, grad_weight=1.0, mono_weight=0.1, te_weight=2.0):
        super(PhysicsGuidedLoss, self).__init__()
        self.dataset = dataset
        self.core_weight = core_weight
        self.grad_weight = grad_weight
        self.mono_weight = mono_weight
        self.te_weight = te_weight
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, output, target):
        
        target_orig = self.dataset.get_original_y(target)
        mask = (target_orig > 0.0).float()
        
        
        batch_size = output.shape[0]
        points = output.shape[1] // 2
        
        
        rho = np.linspace(0, 1, points)
        weight_vec = np.ones(points, dtype=np.float32)
        weight_vec[rho < 0.4] = self.core_weight
        
       
        full_weight_vec = np.concatenate([weight_vec, weight_vec])
        full_weight_tensor = torch.tensor(full_weight_vec, device=output.device).expand(batch_size, -1)
        
        
        squared_err = (output - target) ** 2
        weighted_err = squared_err * full_weight_tensor * mask
        
        valid_count = mask.sum()
        if valid_count > 0:
            mask_ti = mask[:, :points]
            mask_te = mask[:, points:]
            valid_count_ti = mask_ti.sum()
            valid_count_te = mask_te.sum()
            
            mse_loss_ti = (weighted_err[:, :points]).sum() / valid_count_ti if valid_count_ti > 0 else torch.tensor(0.0, device=output.device)
            mse_loss_te = (weighted_err[:, points:]).sum() / valid_count_te if valid_count_te > 0 else torch.tensor(0.0, device=output.device)
        else:
            return torch.tensor(0.0, device=output.device, requires_grad=True), valid_count, torch.tensor(0.0)

        
        ti_pred = output[:, :points]
        te_pred = output[:, points:]
        ti_true = target[:, :points]
        te_true = target[:, points:]
        
        ti_grad_pred = ti_pred[:, 1:] - ti_pred[:, :-1]
        te_grad_pred = te_pred[:, 1:] - te_pred[:, :-1]
        ti_grad_true = ti_true[:, 1:] - ti_true[:, :-1]
        te_grad_true = te_true[:, 1:] - te_true[:, :-1]
        
        
        ti_grad_mask = mask_ti[:, 1:] * mask_ti[:, :-1]
        te_grad_mask = mask_te[:, 1:] * mask_te[:, :-1]
        
        grad_loss_ti = ((ti_grad_pred - ti_grad_true)**2 * ti_grad_mask).sum() / (ti_grad_mask.sum() + 1e-8)
        grad_loss_te = ((te_grad_pred - te_grad_true)**2 * te_grad_mask).sum() / (te_grad_mask.sum() + 1e-8)

        
        mono_penalty_ti = (torch.relu(ti_grad_pred) * ti_grad_mask).sum() / (ti_grad_mask.sum() + 1e-8)
        mono_penalty_te = (torch.relu(te_grad_pred) * te_grad_mask).sum() / (te_grad_mask.sum() + 1e-8)
        
        
        loss_ti = mse_loss_ti + self.grad_weight * grad_loss_ti + self.mono_weight * mono_penalty_ti
        loss_te = mse_loss_te + self.grad_weight * grad_loss_te + self.mono_weight * mono_penalty_te
        
        
        total_loss = loss_ti + self.te_weight * loss_te
        
        
        pure_mse = (squared_err * mask).sum()
        
        return total_loss, valid_count, pure_mse


class ResidualBlock(nn.Module):
   
    def __init__(self, hidden_dim, dropout_prob=0.2):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob)
        )

    def forward(self, x):
        return x + self.block(x)

class ResMLP(nn.Module):
    
    def __init__(self, input_dim, output_dim, hidden_dim=256, num_blocks=3, dropout_prob=0.2):
        super(ResMLP, self).__init__()
        
       
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob)
        )
        
        
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout_prob) for _ in range(num_blocks)]
        )
        
        
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.input_proj(x)
        x = self.res_blocks(x)
        return self.output_proj(x)

class CNNProfileGenerator(nn.Module):
   
    def __init__(self, input_dim, output_dim, hidden_dim=128, kernel_size=5):
        super(CNNProfileGenerator, self).__init__()
        self.output_dim = output_dim
        
        
        self.fc_in = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, output_dim * 4) # Project to a large vector
        )
        
       
        self.conv_layers = nn.Sequential(
            nn.Conv1d(4, 16, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 16, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Conv1d(16, 1, kernel_size=1) # Project to 1 channel (final value)
        )
        
    def forward(self, x):
        
        x = self.fc_in(x) 
       
        x = x.view(-1, 4, self.output_dim)
        
        
        x = self.conv_layers(x) 
        
       
        return x.squeeze(1)

class TransformerPredictor(nn.Module):
    
    def __init__(self, input_dim, output_dim, embed_dim=64, num_heads=4, num_layers=2):
        super(TransformerPredictor, self).__init__()
        
        
        self.input_proj = nn.Linear(input_dim, embed_dim)
        
       
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        
        self.output_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x):
        x_emb = self.input_proj(x)
        x_emb = x_emb.unsqueeze(1) 
        x_out = self.transformer_encoder(x_emb)
        x_out = x_out.squeeze(1)
        return self.output_head(x_out)

class SpatialTransformerPredictor(nn.Module):
   
    def __init__(self, macro_dim, spec_dim=128, output_dim=64, embed_dim=128, num_heads=8, num_layers=4):
        super(SpatialTransformerPredictor, self).__init__()
        
        assert output_dim % 2 == 0, "Output dim must be even (Ti + Te)"
        self.num_spatial_points = output_dim // 2
        
        
        self.macro_attention = MacroFeatureAttention(input_dim=macro_dim)
        
        
        self.spec_encoder = Spec2DEncoder(output_dim=spec_dim)
        
        
        fused_dim = macro_dim + spec_dim
        self.input_proj = nn.Sequential(
            nn.Linear(fused_dim, embed_dim * self.num_spatial_points),
            nn.GELU()
        )
        
        
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_spatial_points, embed_dim))
        
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4, 
            batch_first=True,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        
        self.ti_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
        self.te_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, macro_x, spec_x):
        batch_size = macro_x.size(0)
        
        
        attended_macro_x, macro_weights = self.macro_attention(macro_x)
        
        
        spec_emb = self.spec_encoder(spec_x)
        fused_x = torch.cat([attended_macro_x, spec_emb], dim=1)
        
        
        x_spatial = self.input_proj(fused_x).view(batch_size, self.num_spatial_points, -1)
        
        
        x_spatial = x_spatial + self.pos_embedding
        
        
        x_encoded = self.transformer_encoder(x_spatial)
        
        
        ti_pred = self.ti_head(x_encoded).squeeze(-1) 
        te_pred = self.te_head(x_encoded).squeeze(-1) 
        
        
        return torch.cat([ti_pred, te_pred], dim=1), macro_weights

class ConvMLPModel(nn.Module):
   
    def __init__(self, input_dim, output_dim, hidden_dim=256, dropout_prob=0.2):
        super(ConvMLPModel, self).__init__()
        
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.gelu1 = nn.GELU()
        
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.gelu2 = nn.GELU()
        
       
        flattened_size = 32 * input_dim
        
        
        self.mlp = nn.Sequential(
            nn.Linear(flattened_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        
        x = x.unsqueeze(1)
        
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.gelu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.gelu2(x)
        
        
        x = x.view(x.size(0), -1)
        
        
        output = self.mlp(x)
        return output

class DualConvMLPModel(nn.Module):
    
    def __init__(self, input_dim, output_dim=32, hidden_dim=256, dropout_prob=0.2):
        super(DualConvMLPModel, self).__init__()
        
        self.model_ti = ConvMLPModel(input_dim, output_dim, hidden_dim, dropout_prob)
        self.model_te = ConvMLPModel(input_dim, output_dim, hidden_dim, dropout_prob)

    def forward(self, x):
        out_ti = self.model_ti(x)
        out_te = self.model_te(x)
        
        return torch.cat([out_ti, out_te], dim=1)


def main():
    
    parser = argparse.ArgumentParser(description="Train Tokamak Profile Prediction Model")
    parser.add_argument('--gpu', type=str, default=None, help='Specify GPU ID to use (e.g., "0" or "1"). Defaults to auto-detect.')
    args = parser.parse_args()

    set_seed(SEED)
    
    
    if args.gpu is not None:
        
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        print(f"Force using GPU(s): {args.gpu}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"Current CUDA device ID: {torch.cuda.current_device()}")
        print(f"Current CUDA device name: {torch.cuda.get_device_name(device)}")
    
    print("Initializing Dataset...")
    dataset = TokamakDataset(DATA_DIR)
    
    if len(dataset) == 0:
        print("No data found. Exiting.")
        return

    
    total_size = len(dataset)
    test_size = max(16, int(0.1 * total_size)) 
    val_size = int(0.2 * total_size)
    train_size = total_size - test_size - val_size
    
    print(f"Data Split - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    
    indices = list(range(total_size))
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
   
    print("Computing normalization statistics on Training Set...")
    
    train_x_macro = dataset.X_macro[train_indices]
    train_x_spec = dataset.X_spec[train_indices]
    train_y = dataset.Y_data[train_indices]
    
    x_macro_mean = train_x_macro.mean(dim=0)
    x_macro_std = train_x_macro.std(dim=0)
    
    
    x_spec_mean = train_x_spec.mean()
    x_spec_std = train_x_spec.std()
    
    y_mean = train_y.mean(dim=0)
    y_std = train_y.std(dim=0)
    
    
    dataset.set_normalization(x_macro_mean, x_macro_std, x_spec_mean, x_spec_std, y_mean, y_std)
    
    
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    
    print("Visualizing Interpolation...")
    visualize_interpolation(dataset, save_name='interpolation_raw_vs_smooth_attn.png')
    
    
    try:
        example_macro, example_spec, example_y, _ = next(iter(train_loader))
        print(f"\nBatch Shapes:")
        print(f"Input Macro: {example_macro.shape}")
        print(f"Input Spec (2D): {example_spec.shape}")
        print(f"Output (Ti + Te Profiles): {example_y.shape}")
        
        macro_dim = example_macro.shape[1]
        output_dim = example_y.shape[1]
        
        MODEL_TYPE = 'SpatialTransformer'
        
        if MODEL_TYPE == 'SpatialTransformer':
            print(f"\nInitializing SpatialTransformerPredictor with Macro Dim: {macro_dim}, Output Dim: {output_dim}")
            model = SpatialTransformerPredictor(macro_dim=macro_dim, spec_dim=128, output_dim=output_dim, embed_dim=128, num_heads=8, num_layers=4)
        else:
            raise ValueError(f"Only SpatialTransformer is supported for dual inputs currently. Got: {MODEL_TYPE}")
        
        
        model = model.to(device)
        output, attn_weights = model(example_macro.to(device), example_spec.to(device))
        print(f"Model Output Shape: {output.shape}, Attention Weights Shape: {attn_weights.shape}")
        print("Forward pass successful.")
        
        
        criterion = PhysicsGuidedLoss(dataset, core_weight=3.0, grad_weight=0.5, mono_weight=0.1, te_weight=2.0)
        print(f"Using Physics-Guided Loss Function with te_weight={criterion.te_weight}.")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30, verbose=True)
        
        num_epochs = 400
        print(f"\nStarting training for {num_epochs} epochs...")
        
        best_val_mse = float('inf')
        save_path = 'best_model_attention.pth'
        
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            running_mse = 0.0
            total_elements = 0
            
            for macro_inputs, spec_inputs, targets, _ in train_loader:
                macro_inputs = macro_inputs.to(device)
                spec_inputs = spec_inputs.to(device)
                targets = targets.to(device)
                
               
                optimizer.zero_grad()
                
                
                outputs, _ = model(macro_inputs, spec_inputs)
                
                
                loss, valid_count, batch_mse = criterion(outputs, targets)
                
                
                loss.backward()
                
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                running_loss += loss.item() * valid_count.item()
                running_mse += batch_mse.item()
                total_elements += valid_count.item()
            
           
            epoch_train_loss = running_loss / total_elements if total_elements > 0 else 0.0
            epoch_train_mse = running_mse / total_elements if total_elements > 0 else 0.0
            
            
            val_loss, val_mae, val_rmse, val_r2 = evaluate_model(model, val_loader, nn.MSELoss(), dataset, device=device)
            
            
            scheduler.step(val_loss)
            
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] - Train Total Loss: {epoch_train_loss:.6f} - Train MSE: {epoch_train_mse:.6f} - Val MSE: {val_loss:.6f} - Val R2: {val_r2:.6f}")
            
           
            if val_loss < best_val_mse:
                best_val_mse = val_loss
                torch.save(model.state_dict(), save_path)
                
        
        print("\nTraining finished.")
        print(f"Best Validation Loss: {best_val_mse:.6f}")
        
        
        print("\nLoading best model for testing...")
        model.load_state_dict(torch.load(save_path, map_location=device))
        
        test_loss, test_mae, test_rmse, test_r2 = evaluate_model(model, test_loader, criterion, dataset, device=device)
        print(f"Final Test Results - Loss: {test_loss:.6f} - RMSE: {test_rmse:.6f} - MAE: {test_mae:.6f} - R2: {test_r2:.6f}")
        
        
        visualize_results(model, test_loader, dataset, save_name='test_predictions_attn.png', device=device)
        
        
        visualize_attention_weights(model, test_loader, dataset, save_name='feature_importance_attn.png', device=device)
        
    except Exception as e:
        print(f"An error occurred during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
