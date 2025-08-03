import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
from torch.optim import LBFGS, Adam
from tqdm import tqdm
import scipy.io
#from sample import sample
from utils import *
import torch
import numpy as np
import time
import os
import datetime 
import matplotlib.pyplot as plt

from utils.datasets import get_Dataset
from utils.sample import sample
from utils.loss import neumann_boundary_loss,water_boundary_loss,constant_temperature_boundary_loss,Continuity_Loss,pde_loss,analytical_loss,init_loss
from utils.util import get_n_params,init_weights
from model_dict import get_model
# 先做2D的可视化结果


def set_z_t(x,z,t):
    z = torch.full_like(x, fill_value=5)  
    t = torch.full_like(x, fill_value=5)  
    return z, t


def predict_temperature(checkpoint_path, device, args):
    data = get_Dataset(device)
    D1, D2, D3 = data.sample_pde()
    I1, I2, I3 = data.sample_interface()
    Bottom,Left,Right,Top,Front,Back = data.sample_boundary()
    # CuCrZr\Cu\W
    x_CuCrZr,y_CuCrZr,z_CuCrZr,t_CuCrZr= data.deal_data(D1)
    x_Cu,y_Cu,z_Cu,t_Cu= data.deal_data(D2)
    x_W,y_W,z_W,t_W= data.deal_data(D3)
    # interface
    x_I1,y_I1,z_I1,t_I1 = data.deal_data(I1)
    x_I2,y_I2,z_I2,t_I2 = data.deal_data(I2)
    x_I3,y_I3,z_I3,t_I3 = data.deal_data(I3)
    # left\right\front\bottom\top
    x_Left,y_Left,z_Left,t_Left = data.deal_data(Left)
    x_Right,y_Right,z_Right,t_Right = data.deal_data(Right)
    x_Front,y_Front,z_Front,t_Front = data.deal_data(Front)
    x_Back,y_Back,z_Back,t_Back = data.deal_data(Back)
    x_Bottom,y_Bottom,z_Bottom,t_Bottom = data.deal_data(Bottom)
    x_Top,y_Top,z_Top,t_Top = data.deal_data(Top)
    
    # 检查检查点文件是否存在
    if not os.path.exists(checkpoint_path):
        return 
    print(f"Loading...: {checkpoint_path}")
    
    # ------------------------------------------------------------模型选择---------------------------------------------
    if args.model == 'QRes':
        model1 = get_model(args).Model(in_dim=4, hidden_dim=512, out_dim=1, num_layer=4).to(device)
        model2 = get_model(args).Model(in_dim=4, hidden_dim=512, out_dim=1, num_layer=4).to(device)
        model3 = get_model(args).Model(in_dim=4, hidden_dim=512, out_dim=1, num_layer=4).to(device)
        model1.apply(init_weights)
        model2.apply(init_weights)
        model3.apply(init_weights)
    elif args.model == 'PINNsFormer': 
        model1 = get_model(args).Model(in_dim=4, hidden_dim=32, out_dim=1, num_layer=1).to(device)
        model2 = get_model(args).Model(in_dim=4, hidden_dim=32, out_dim=1, num_layer=1).to(device)
        model3 = get_model(args).Model(in_dim=4, hidden_dim=32, out_dim=1, num_layer=1).to(device)
        model1.apply(init_weights)
        model2.apply(init_weights)
        model3.apply(init_weights)
    elif args.model == 'PINNMamba': 
        model1 = get_model(args).Model(in_dim=4, hidden_dim=32, out_dim=1, num_layer=1).to(device)
        model2 = get_model(args).Model(in_dim=4, hidden_dim=32, out_dim=1, num_layer=1).to(device)
        model3 = get_model(args).Model(in_dim=4, hidden_dim=32, out_dim=1, num_layer=1).to(device)
        model1.apply(init_weights)
        model2.apply(init_weights)
        model3.apply(init_weights)
    elif args.model == 'PirateNet':
        model1 =get_model(args).Model(input_dim=4, hidden_dim=128, output_dim=1, num_layers=4, 
                        activation='tanh', nonlinearity=0.0,
                        fourier_emb={'embed_scale': 1.0, 'embed_dim': 256}).to(device)
        model2 = get_model(args).Model(input_dim=4, hidden_dim=128, output_dim=1, num_layers=4,
                        activation='tanh', nonlinearity=0.0,
                        fourier_emb={'embed_scale': 1.0, 'embed_dim': 256}).to(device)
        model3 = get_model(args).Model(input_dim=4, hidden_dim=128, output_dim=1, num_layers=4,
                        activation='tanh', nonlinearity=0.0,
                        fourier_emb={'embed_scale': 1.0, 'embed_dim': 256}).to(device) 
        model1.apply(init_weights)
        model2.apply(init_weights)
        model3.apply(init_weights)
    elif args.model == 'FLS':
        model1 = get_model(args).Model(in_dim=4, hidden_dim=256, out_dim=1, num_layer=4).to(device)
        model2 = get_model(args).Model(in_dim=4, hidden_dim=256, out_dim=1, num_layer=4).to(device)
        model3 = get_model(args).Model(in_dim=4, hidden_dim=256, out_dim=1, num_layer=4).to(device)
        model1.apply(init_weights)
        model2.apply(init_weights)
        model3.apply(init_weights)
    else:
        model1 = get_model(args).Model(in_dim=4, hidden_dim=256, out_dim=1, num_layer=4).to(device)
        model2 = get_model(args).Model(in_dim=4, hidden_dim=256, out_dim=1, num_layer=4).to(device)
        model3 = get_model(args).Model(in_dim=4, hidden_dim=256, out_dim=1, num_layer=4).to(device)
        model1.apply(init_weights)
        model2.apply(init_weights)
        model3.apply(init_weights)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model1.load_state_dict(checkpoint['model1_state_dict'])
    model2.load_state_dict(checkpoint['model2_state_dict'])
    model3.load_state_dict(checkpoint['model3_state_dict'])
    
    data = get_Dataset(device)
    D1, D2, D3 = data.sample_pde()
    I1, I2, I3 = data.sample_interface()
    Bottom,Left,Right,Top,Front,Back = data.sample_boundary()

    x_CuCrZr,y_CuCrZr,z_CuCrZr,t_CuCrZr= data.deal_data(D1)
    x_Cu,y_Cu,z_Cu,t_Cu= data.deal_data(D2)
    x_W,y_W,z_W,t_W= data.deal_data(D3)

    _,t_CuCrZr = set_z_t(y_CuCrZr,z_CuCrZr,t_CuCrZr)
    _,t_Cu = set_z_t(y_Cu,z_Cu,t_Cu)
    _,t_W = set_z_t(y_W,z_W,t_W)
    if args.model == 'KAN':
        with torch.no_grad():
            predict_W = model3(x_W,y_W,z_W,t_W)
            predict_Cu = model2(x_Cu,y_Cu,z_Cu,t_Cu)
            predict_CuCrZr = model1(x_CuCrZr,y_CuCrZr,z_CuCrZr,t_CuCrZr)
    else:
        model1.eval()
        model2.eval()
        model3.eval()
        predict_W = model3(x_W,y_W,z_W,t_W)
        predict_Cu = model2(x_Cu,y_Cu,z_Cu,t_Cu)
        predict_CuCrZr = model1(x_CuCrZr,y_CuCrZr,z_CuCrZr,t_CuCrZr)
    print(predict_W.shape)

    x_combined = torch.cat([x_W, x_Cu, x_CuCrZr], dim=0)
    y_combined = torch.cat([y_W, y_Cu, y_CuCrZr], dim=0)
    z_combined = torch.cat([z_W, z_Cu, z_CuCrZr], dim=0)
    T_combined = torch.cat([predict_W, predict_Cu, predict_CuCrZr], dim=0)

    print(predict_W.max(),predict_W.min())
    print(predict_Cu.max(),predict_Cu.min())
    print(predict_CuCrZr.max(),predict_CuCrZr.min())
    import matplotlib.pyplot as plt
    
    def plot_temperature_distribution(x, y, z, temps, material_name):
        if isinstance(x, torch.Tensor):
            x = x.cpu().detach().numpy()
            y = y.cpu().detach().numpy() 
            z = z.cpu().detach().numpy()
            temp = temps.cpu().detach().numpy()
        else:
            temp = temps

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(x, z, y, c=temp, cmap='jet', s=10, alpha=0.6)
        fig.colorbar(scatter, ax=ax, label='Temperature (°C)')
        
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel('Z Coordinate')
        ax.set_title(f'3D Temperature Distribution - {material_name}')
        
        for i, angle in enumerate([120]):
            ax.view_init(elev=20, azim=angle)
            plt.savefig(f'pic/temp3d_{material_name}_view{i+1}.png')
        plt.close()

    
    plot_temperature_distribution(x_CuCrZr, y_CuCrZr, z_CuCrZr, predict_CuCrZr, 'CuCrZr')
    plot_temperature_distribution(x_Cu, y_Cu, z_Cu, predict_Cu, 'Cu') 
    plot_temperature_distribution(x_W, y_W, z_W, predict_W, 'W')
    
def get_MAE(checkpoint_path, device, args):
    list_MAE = []
    for i in range(5,6):
        select_time = i
        print(select_time)
        data = np.load("data/data_analysis.npy", allow_pickle=True).item()
        W = data['W']
        Cu = data['Cu']
        CuCrZr =data['CuCrZr']
        def deal_dataA(domain):
            factor = 1000
            if isinstance(domain, dict):
                x = np.array(domain['x'], dtype=np.float32)
                y = np.array(domain['y'], dtype=np.float32)
                z = np.array(domain['z'], dtype=np.float32)
                t = np.array(domain['t'], dtype=np.float32)
                T = np.array(domain['temperature'], dtype=np.float32)      
                x = x.reshape(-1, 1)
                y = y.reshape(-1, 1)
                z = z.reshape(-1, 1)
                t = t.reshape(-1, 1)
                T = T.reshape(-1, 1)
            else:
                x = domain[:, 0].reshape(-1, 1).astype(np.float32)
                y = domain[:, 1].reshape(-1, 1).astype(np.float32)
                z = domain[:, 2].reshape(-1, 1).astype(np.float32)
                t = domain[:, 3].reshape(-1, 1).astype(np.float32)
                T = domain[:, 4].reshape(-1, 1).astype(np.float32)

            x = torch.tensor(x*factor, dtype=torch.float32, requires_grad=True).to(device)
            y = torch.tensor(y*factor, dtype=torch.float32, requires_grad=True).to(device)
            z = torch.tensor(z*factor, dtype=torch.float32, requires_grad=True).to(device)
            t = torch.tensor(t, dtype=torch.float32, requires_grad=True).to(device)
            T = torch.tensor(T-273.15, dtype=torch.float32, requires_grad=True).to(device)
            return x,y,z,t,T

        Ax_CuCrZr,Ay_CuCrZr,Az_CuCrZr,At_CuCrZr,T_CuCrZr = deal_dataA(CuCrZr)
        Ax_Cu,Ay_Cu,Az_Cu,At_Cu,T_Cu = deal_dataA(Cu)
        Ax_W,Ay_W,Az_W,At_W,T_W = deal_dataA(W)
        def get_t5_indices(At_tensor, target_t, tolerance=1e-5):
            mask = (torch.abs(At_tensor - target_t) < tolerance).squeeze()
            return torch.where(mask)[0].cpu().numpy()

        indices_CuCrZr = get_t5_indices(At_CuCrZr,target_t=select_time)
        indices_Cu = get_t5_indices(At_Cu,target_t=select_time)
        indices_W = get_t5_indices(At_W,target_t=select_time)

        def extract_t5_data(x, y, z, t, T, indices):
            return (
                x[indices], y[indices], z[indices], t[indices],T[indices]
            )

        # CuCrZr
        x_t5_CuCrZr, y_t5_CuCrZr, z_t5_CuCrZr, t_t5_CuCrZr, T_t5_CuCrZr = extract_t5_data(
            Ax_CuCrZr, Ay_CuCrZr, Az_CuCrZr, At_CuCrZr,T_CuCrZr, indices_CuCrZr
        )

        # Cu
        x_t5_Cu, y_t5_Cu, z_t5_Cu, t_t5_Cu, T_t5_Cu = extract_t5_data(
            Ax_Cu, Ay_Cu, Az_Cu,At_Cu,T_Cu, indices_Cu
        )

        # W
        x_t5_W, y_t5_W, z_t5_W, t_t5_W, T_t5_W = extract_t5_data(
            Ax_W, Ay_W, Az_W, At_W, T_W, indices_W
        )

        if not os.path.exists(checkpoint_path):
            print('NO Checkpoint !')
            return 100000

        print(f"Loading...: {checkpoint_path}")

        if args.model == 'QRes':
            model1 = get_model(args).Model(in_dim=4, hidden_dim=512, out_dim=1, num_layer=4).to(device)
            model2 = get_model(args).Model(in_dim=4, hidden_dim=512, out_dim=1, num_layer=4).to(device)
            model3 = get_model(args).Model(in_dim=4, hidden_dim=512, out_dim=1, num_layer=4).to(device)
            model1.apply(init_weights)
            model2.apply(init_weights)
            model3.apply(init_weights)
        elif args.model == 'PINNsFormer': 
            model1 = get_model(args).Model(in_dim=4, hidden_dim=32, out_dim=1, num_layer=1).to(device)
            model2 = get_model(args).Model(in_dim=4, hidden_dim=32, out_dim=1, num_layer=1).to(device)
            model3 = get_model(args).Model(in_dim=4, hidden_dim=32, out_dim=1, num_layer=1).to(device)
            model1.apply(init_weights)
            model2.apply(init_weights)
            model3.apply(init_weights)
        elif args.model == 'PINNMamba': 
            model1 = get_model(args).Model(in_dim=4, hidden_dim=32, out_dim=1, num_layer=1).to(device)
            model2 = get_model(args).Model(in_dim=4, hidden_dim=32, out_dim=1, num_layer=1).to(device)
            model3 = get_model(args).Model(in_dim=4, hidden_dim=32, out_dim=1, num_layer=1).to(device)
            model1.apply(init_weights)
            model2.apply(init_weights)
            model3.apply(init_weights)
        elif args.model == 'PirateNet':
            model1 =get_model(args).Model(input_dim=4, hidden_dim=128, output_dim=1, num_layers=4, 
                            activation='tanh', nonlinearity=0.0,
                            fourier_emb={'embed_scale': 1.0, 'embed_dim': 256}).to(device)
            model2 = get_model(args).Model(input_dim=4, hidden_dim=128, output_dim=1, num_layers=4,
                            activation='tanh', nonlinearity=0.0,
                            fourier_emb={'embed_scale': 1.0, 'embed_dim': 256}).to(device)
            model3 = get_model(args).Model(input_dim=4, hidden_dim=128, output_dim=1, num_layers=4,
                            activation='tanh', nonlinearity=0.0,
                            fourier_emb={'embed_scale': 1.0, 'embed_dim': 256}).to(device) 
            model1.apply(init_weights)
            model2.apply(init_weights)
            model3.apply(init_weights)
        elif args.model == 'FLS':
            model1 = get_model(args).Model(in_dim=4, hidden_dim=256, out_dim=1, num_layer=4).to(device)
            model2 = get_model(args).Model(in_dim=4, hidden_dim=256, out_dim=1, num_layer=4).to(device)
            model3 = get_model(args).Model(in_dim=4, hidden_dim=256, out_dim=1, num_layer=4).to(device)
            model1.apply(init_weights)
            model2.apply(init_weights)
            model3.apply(init_weights)
        else:
            model1 = get_model(args).Model(in_dim=4, hidden_dim=256, out_dim=1, num_layer=4).to(device)
            model2 = get_model(args).Model(in_dim=4, hidden_dim=256, out_dim=1, num_layer=4).to(device)
            model3 = get_model(args).Model(in_dim=4, hidden_dim=256, out_dim=1, num_layer=4).to(device)
            model1.apply(init_weights)
            model2.apply(init_weights)
            model3.apply(init_weights)

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model1.load_state_dict(checkpoint['model1_state_dict'])
        model2.load_state_dict(checkpoint['model2_state_dict'])
        model3.load_state_dict(checkpoint['model3_state_dict'])


        data = get_Dataset(device)
        D1, D2, D3 = data.sample_pde()
        I1, I2, I3 = data.sample_interface()
        Bottom,Left,Right,Top,Front,Back = data.sample_boundary()
        # CuCrZr\Cu\W
        x_CuCrZr,y_CuCrZr,z_CuCrZr,t_CuCrZr= data.deal_data(D1)
        x_Cu,y_Cu,z_Cu,t_Cu= data.deal_data(D2)
        x_W,y_W,z_W,t_W= data.deal_data(D3)

        _,t_CuCrZr = set_z_t(y_CuCrZr,z_CuCrZr,select_time)
        _,t_Cu = set_z_t(y_Cu,z_Cu,select_time)
        _,t_W = set_z_t(y_W,z_W,select_time)

        if args.model == 'PINNsFormer' or args.model == 'PINNsFormer_Enc_Only':
            model1.eval()
            model2.eval()
            model3.eval()
            x_parts = torch.chunk(x_t5_W, chunks=3, dim=0)  
            y_parts = torch.chunk(y_t5_W, chunks=3, dim=0)
            z_parts = torch.chunk(z_t5_W, chunks=3, dim=0)
            t_parts = torch.chunk(t_t5_W, chunks=3, dim=0)

            group1 = (x_parts[0], y_parts[0], z_parts[0], t_parts[0])
            group2 = (x_parts[1], y_parts[1], z_parts[1], t_parts[1])
            group3 = (x_parts[2], y_parts[2], z_parts[2], t_parts[2])

            predict_W1 = model3(x_parts[0], y_parts[0], z_parts[0], t_parts[0]) 
            predict_W2 = model3(x_parts[1], y_parts[1], z_parts[1], t_parts[1]) 
            predict_W3 = model3(x_parts[2], y_parts[2], z_parts[2], t_parts[2]) 
            predict_W = torch.cat([predict_W1, predict_W2, predict_W3], dim=0)
            predict_Cu = model2(x_t5_Cu, y_t5_Cu, z_t5_Cu, t_t5_Cu)
            predict_CuCrZr = model1(x_t5_CuCrZr, y_t5_CuCrZr, z_t5_CuCrZr, t_t5_CuCrZr)
        elif args.model == 'KAN':
            with torch.no_grad():
                predict_W = model3(x_t5_W, y_t5_W, z_t5_W, t_t5_W)
                predict_Cu = model2(x_t5_Cu, y_t5_Cu, z_t5_Cu, t_t5_Cu)
                predict_CuCrZr = model1(x_t5_CuCrZr, y_t5_CuCrZr, z_t5_CuCrZr, t_t5_CuCrZr)
        else:
            model1.eval()
            model2.eval()
            model3.eval()
            predict_W = model3(x_t5_W, y_t5_W, z_t5_W, t_t5_W)
            predict_Cu = model2(x_t5_Cu, y_t5_Cu, z_t5_Cu, t_t5_Cu)
            predict_CuCrZr = model1(x_t5_CuCrZr, y_t5_CuCrZr, z_t5_CuCrZr, t_t5_CuCrZr)


        x_combined = torch.cat([x_t5_W, x_t5_Cu, x_t5_CuCrZr], dim=0)
        y_combined = torch.cat([y_t5_W, y_t5_Cu, y_t5_CuCrZr], dim=0)
        z_combined = torch.cat([z_t5_W, z_t5_Cu, z_t5_CuCrZr], dim=0)
        t_combined = torch.cat([t_t5_W, t_t5_Cu, t_t5_CuCrZr], dim=0)
        gt_combined = torch.cat([T_t5_W, T_t5_Cu, T_t5_CuCrZr], dim=0)
        T_combined = torch.cat([predict_W, predict_Cu, predict_CuCrZr], dim=0)


        def plot_temperature_distribution(x, y, z, temps, material_name):
            if isinstance(x, torch.Tensor):
                x = x.cpu().detach().numpy()
                y = y.cpu().detach().numpy() 
                z = z.cpu().detach().numpy()
                temp = temps.cpu().detach().numpy()
            else:
                temp = temps

            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            scatter = ax.scatter(x, z, y, c=temp, cmap='jet', s=10, alpha=0.6)
            fig.colorbar(scatter, ax=ax, label='Temperature (°C)')
            
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.set_zlabel('Z Coordinate')
            ax.set_title(f'3D Temperature Distribution - {material_name}')
            
            # 保存多个视角
            for i, angle in enumerate([120]):
                ax.view_init(elev=20, azim=angle)
                plt.savefig(f'pic/temp3d_{material_name}_view{i+1}.png')
            plt.close()

        #plot_temperature_distribution(x_t5_CuCrZr, y_t5_CuCrZr, z_t5_CuCrZr, predict_CuCrZr, f'CuCrZr_{select_time}s')
        #plot_temperature_distribution(x_t5_Cu, y_t5_Cu, z_t5_Cu, predict_Cu, f'Cu_{select_time}s') 
        #plot_temperature_distribution(x_t5_W, y_t5_W, z_t5_W, predict_W, f'W_{select_time}s')
        plot_temperature_distribution(x_combined, y_combined, z_combined, gt_combined, f'GT_{select_time}s')
        plot_temperature_distribution(x_combined, y_combined, z_combined, T_combined, f'ALL_{select_time}s')
        plot_temperature_distribution(x_combined, y_combined, z_combined, abs(T_combined-gt_combined), f'abs_{select_time}')
        mae_CuCrZr = torch.abs(T_t5_CuCrZr-predict_CuCrZr).mean()  
        mae_Cu = torch.abs(T_t5_Cu-predict_Cu).mean()  
        mae_W = torch.abs(T_t5_W-predict_W).mean()  
        print(f'mae_CuCrZr:{mae_CuCrZr}')
        print(f'mae_Cu:{mae_Cu}')
        print(f'mae_W:{mae_W}')
        if len(T_combined.shape) == 3:
            T_combined = T_combined.squeeze(1)
        mae = torch.abs(T_combined - gt_combined).mean()  
        print("Mean Absolute Error (MAE):", mae.item()) 
    #return mae

        """
        abs_error = torch.abs(T_combined - gt_combined)
        mask = abs_error > 50

        x_outlier = x_combined[mask]
        y_outlier = y_combined[mask]
        z_outlier = z_combined[mask]
        t_outlier = t_combined[mask]
        gt_outlier = gt_combined[mask]
        error_outlier = abs_error[mask]
        radius = torch.sqrt(x_outlier**2 + y_outlier**2)
        mask_CuCrZr = radius < 7.5
        mask_Cu = (radius >= 7.5) & (radius < 10.5)
        mask_W = radius >= 10.5
        

        data_dict = {
            'CuCrZr': {
                'x': x_outlier[mask_CuCrZr].detach().cpu().numpy(),
                'y': y_outlier[mask_CuCrZr].detach().cpu().numpy(),
                'z': z_outlier[mask_CuCrZr].detach().cpu().numpy(),
                't': t_outlier[mask_CuCrZr].detach().cpu().numpy(),
                'temperature': gt_outlier[mask_CuCrZr].detach().cpu().numpy()
            },
            'Cu': {
                'x': x_outlier[mask_Cu].detach().cpu().numpy(),
                'y': y_outlier[mask_Cu].detach().cpu().numpy(),
                'z': z_outlier[mask_Cu].detach().cpu().numpy(),
                't': t_outlier[mask_Cu].detach().cpu().numpy(),
                'temperature': gt_outlier[mask_Cu].detach().cpu().numpy()
            },
            'W': {
                'x': x_outlier[mask_W].detach().cpu().numpy(),
                'y': y_outlier[mask_W].detach().cpu().numpy(),
                'z': z_outlier[mask_W].detach().cpu().numpy(),
                't': t_outlier[mask_W].detach().cpu().numpy(),
                'temperature': gt_outlier[mask_W].detach().cpu().numpy()
            }
        }

        np.save('sample2.npy', data_dict)
        print("sample2.npy")
        print(len(z_outlier))
        # 绘制这些点
        def plot_outlier_points(x, y, z, error, material_name):
            if isinstance(x, torch.Tensor):
                x = x.cpu().detach().numpy()
                y = y.cpu().detach().numpy()
                z = z.cpu().detach().numpy()
                error = error.cpu().detach().numpy()
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(x, z, y, c=error, cmap='hot', s=20, alpha=0.8)
            fig.colorbar(scatter, ax=ax, label='Absolute Error (°C)')
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.set_zlabel('Z Coordinate')
            ax.set_title(f'Outlier Points (|Error|>50) - {material_name}')
            plt.savefig(f'pic/outlier_points_{material_name}.png')
            plt.close()

        plot_outlier_points(x_outlier, y_outlier, z_outlier, error_outlier, f'ALL_{select_time}s')
        """