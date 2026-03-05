import os
import matplotlib.pyplot as plt
import numpy as np

dir = './visualizations'
os.makedirs(dir, exist_ok=True)

# channel_names = ['FlowVelocity_X', 'FlowVelocity_Y', 'Pressure', 'Density']  # ns
channel_names = ['Flow Velocity X', 'Flow Velocity Y', 'mask']  # cfdbench
# channel_names = ['Flow Velocity X', 'Flow Velocity Y', 'div', 'vor','press' ]  # pdearena


def visualize_sample(pred, target,i,dataset_name, sample_idx=0):
    """
    pred: torch.Tensor, [B, T, H, W, C] 或 [B, H, W,t, C]
    target: torch.Tensor, [B, T, H, W, C] 或 [B, H, W, C]
    sample_idx: 可视化 batch 中第几个样本
    """
    dir = './visualizations/prediction'
    if not os.path.exists(dir):
        os.makedirs(dir)
    print(i)
    dataset_dir=os.path.join(dir,dataset_name)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    save_dir=os.path.join(dataset_dir,str(i))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if pred.ndim == 5:
        pred_s = pred[sample_idx, :,:,0,:].cpu().numpy()  # 第一个时间步
        target_s = target[sample_idx, :,:,0,:].cpu().numpy()
    elif pred.ndim == 4:
        pred_s = pred[sample_idx].cpu().numpy()
        target_s = target[sample_idx].cpu().numpy()
    else:
        raise ValueError("pred shape not supported")

    n_channels = pred_s.shape[-1]


    if len(channel_names) < n_channels:
        auto_names = [f'Channel {k+1}' for k in range(n_channels - len(channel_names))]
        names = channel_names + auto_names
    else:
        names = channel_names[:n_channels]

    for c in range(n_channels):
        channel_name = names[c]

        # 真实值
        plt.figure(figsize=(8,6))
        plt.imshow(target_s[..., c], cmap='coolwarm')
        plt.colorbar()
        plt.title(f'{channel_name} - True')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{str(i)}_{channel_name}_true.png'), dpi=300)
        plt.close()

        # 预测值
        plt.figure(figsize=(8,6))
        plt.imshow(pred_s[..., c], cmap='coolwarm')
        plt.colorbar()
        plt.title(f'{channel_name} - Pred')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{str(i)}_{channel_name}_pred.png'), dpi=300)
        plt.close()

        # 误差
        error = pred_s[..., c] - target_s[..., c]
        plt.figure(figsize=(8,6))

        max_error = np.max(np.abs(error))

        v_range = max_error * 2
        plt.imshow(error, cmap='coolwarm', vmin=-v_range, vmax=v_range)
        plt.colorbar()
        plt.title(f'{channel_name} - Error')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{str(i)}_{channel_name}_error.png'), dpi=300)
        plt.close()

    print(f"Visualization saved for sample {str(i)} in {save_dir}")

