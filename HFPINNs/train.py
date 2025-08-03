import numpy as np
import torch
import torch.nn as nn
from torch.optim import LBFGS, Adam
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import time
import os
import datetime 
import argparse

from utils import *
from utils.datasets import get_Dataset
#from utils.model import MLP,PINNsformer,QRes,FLS
from utils.sample import sample
from utils.loss import continues_Q,neumann_boundary_loss,water_boundary_loss,constant_temperature_boundary_loss,Continuity_Loss,pde_loss,analytical_loss,init_loss
from utils.util import get_n_params,init_weights,make_time_sequence,rengion_sample
#from utils.anadata import sample_A

from model_dict import get_model
from visible import predict_temperature,get_MAE
#from visible_FLS_MAE import get_MAE

from conflictfree.grad_operator import ConFIG_update
from conflictfree.utils import get_gradient_vector,apply_gradient_vector

#torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
step_size = 1e-2
num_step = 5
seq_diff = int(1e-2/step_size)

parser = argparse.ArgumentParser('Training Point Optimization')
parser.add_argument('--model', type=str, default='pinn')
parser.add_argument('--device', type=str, default='cuda:0')
args = parser.parse_args()
device = args.device


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


# -----------------------------------train------------------------------------------------
def train():
    pbar = tqdm(range(30000))
    loss_track = []  
    best_loss = float('inf')  
    best_MAE = float('inf')  
    sample(1000)
    
    # 优化器
    optim1 = Adam(model1.parameters(), lr=0.001)
    optim2 = Adam(model2.parameters(), lr=0.001)
    optim3 = Adam(model3.parameters(), lr=0.001)

    # 动态学习率
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optim1, T_max=30000, eta_min=1e-6)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optim2, T_max=30000, eta_min=1e-6)
    scheduler3 = torch.optim.lr_scheduler.CosineAnnealingLR(optim3, T_max=30000, eta_min=1e-6)

    # 开始训练
    for i in pbar:
        #optim.zero_grad()
        optim1.zero_grad()
        optim2.zero_grad()
        optim3.zero_grad()
        #-------------------------------------- 数据读入--------------------------------------------------------
        # 系数设置对应 W/Cu/CuCrZr
        k2, k3, k4 = 173, 403, 318
        rho2, rho3, rho4 = 19298, 8960, 8920
        cp2, cp3, cp4 = 129, 390, 388
        k = 318
        h = 29097
        gradient_list_overall = []
        gradient_list_temp = []
        gradient_variance = 1
        gradient_list_overall1 = []
        gradient_list_temp1 = []
        gradient_variance1 = 1
        gradient_list_overall2 = []
        gradient_list_temp2 = []
        gradient_variance2 = 1

        # 数据获取
        data = get_Dataset(device)
        D1, D2, D3 = data.sample_pde()
        I1, I2, I3 = data.sample_interface()
        Bottom,Left,Right,Top,Front,Back = data.sample_boundary()
        W,Cu,CuCrZr = data.get_analytical_solutions()
        W2,Cu2,CuCrZr2 = data.get_analytical_solutions2()
        Init_W,Init_Cu,Init_CuCrZr = data.get_init()
        # data_all =  data.get_analytical_solutionsALL()
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
        # 解析解
        Ax_CuCrZr,Ay_CuCrZr,Az_CuCrZr,At_CuCrZr,T_CuCrZr = data.deal_dataA(CuCrZr)
        Ax_Cu,Ay_Cu,Az_Cu,At_Cu,T_Cu = data.deal_dataA(Cu)
        Ax_W,Ay_W,Az_W,At_W,T_W = data.deal_dataA(W)
        Ax_CuCrZr2,Ay_CuCrZr2,Az_CuCrZr2,At_CuCrZr2,T_CuCrZr2 = data.deal_dataA(CuCrZr2)
        Ax_Cu2,Ay_Cu2,Az_Cu2,At_Cu2,T_Cu2 = data.deal_dataA(Cu2)
        Ax_W2,Ay_W2,Az_W2,At_W2,T_W2 = data.deal_dataA(W2)
        # init(初始采样点)
        Ix_CuCrZr,Iy_CuCrZr,Iz_CuCrZr,It_CuCrZr,IT_CuCrZr = data.deal_dataA(Init_CuCrZr)
        Ix_Cu,Iy_Cu,Iz_Cu,It_Cu,IT_Cu = data.deal_dataA(Init_Cu)
        Ix_W,Iy_W,Iz_W,It_W,IT_W = data.deal_dataA(Init_W)


        # ------------------------------------------------------构造时间序列----------------------

        if args.model == 'PINNsFormer' or args.model == 'PINNMamba' or args.model == 'slstm' or args.model == 'mlstm' or args.model == 'slstm_block' or args.model == 'mlstm_block' or args.model == 'xlstm_block':
            # CuCrZr\Cu\W
            x_CuCrZr,y_CuCrZr,z_CuCrZr,t_CuCrZr= data.deal_data_seq(D1,num_step=num_step, step=step_size)
            x_Cu,y_Cu,z_Cu,t_Cu= data.deal_data_seq(D2,num_step=num_step, step=step_size)
            x_W,y_W,z_W,t_W= data.deal_data_seq(D3,num_step=num_step, step=step_size)
            # interface
            x_I1,y_I1,z_I1,t_I1 = data.deal_data_seq(I1,num_step=num_step, step=step_size)
            x_I2,y_I2,z_I2,t_I2 = data.deal_data_seq(I2,num_step=num_step, step=step_size)
            x_I3,y_I3,z_I3,t_I3 = data.deal_data_seq(I3,num_step=num_step, step=step_size)
            # left\right\front\bottom\top
            x_Left,y_Left,z_Left,t_Left = data.deal_data_seq(Left,num_step=num_step, step=step_size)
            x_Right,y_Right,z_Right,t_Right = data.deal_data_seq(Right,num_step=num_step, step=step_size)
            x_Front,y_Front,z_Front,t_Front = data.deal_data_seq(Front,num_step=num_step, step=step_size)
            x_Back,y_Back,z_Back,t_Back = data.deal_data_seq(Back,num_step=num_step, step=step_size)
            x_Bottom,y_Bottom,z_Bottom,t_Bottom = data.deal_data_seq(Bottom,num_step=num_step, step=step_size)
            x_Top,y_Top,z_Top,t_Top = data.deal_data_seq(Top,num_step=num_step, step=step_size)

        #-----------------------------------Set-------------------------------------------------
        x_CuCrZr,y_CuCrZr,z_CuCrZr,t_CuCrZr= data.deal_data(D1)
        x_CuCrZr,y_CuCrZr,z_CuCrZr,t_CuCrZr=rengion_sample(x_CuCrZr,y_CuCrZr,z_CuCrZr,t_CuCrZr,sample_num=3,gradient_variance=gradient_variance)
        x_Cu,y_Cu,z_Cu,t_Cu= data.deal_data(D2)
        x_Cu,y_Cu,z_Cu,t_Cu=rengion_sample(x_Cu,y_Cu,z_Cu,t_Cu,sample_num=3,gradient_variance=gradient_variance1)
        x_W,y_W,z_W,t_W= data.deal_data(D3)
        x_W,y_W,z_W,t_W=rengion_sample(x_W,y_W,z_W,t_W,sample_num=3,gradient_variance=gradient_variance2)
        #-----------------------------------------------------------------损失------------------------------------
        # PDE损失(注意这里的参数对应关系)
        loss_CuCrZr,T1 = pde_loss(model1, x_CuCrZr,y_CuCrZr,z_CuCrZr,t_CuCrZr, k4, rho4, cp4)
        loss_Cu,T2 = pde_loss(model2, x_Cu,y_Cu,z_Cu,t_Cu, k3, rho3, cp3)
        loss_W,T3 = pde_loss(model3, x_W,y_W,z_W,t_W, k2, rho2, cp2)
        loss_PDE = loss_CuCrZr + loss_Cu + loss_W
            

        # 边界损失
        loss_bottom = neumann_boundary_loss(model3, x_Bottom, y_Bottom, z_Bottom, t_Bottom , "Y")
        loss_right = neumann_boundary_loss(model3, x_Right, y_Right, z_Right, t_Right , "X")
        loss_left = neumann_boundary_loss(model3, x_Left, y_Left, z_Left, t_Left, "X")
        loss_front = neumann_boundary_loss(model3, x_Front, y_Front, z_Front, t_Front, "Z")
        loss_back = neumann_boundary_loss(model3, x_Back, y_Back, z_Back, t_Back, "Z")
        loss_top = constant_temperature_boundary_loss(model3, x_Top, y_Top, z_Top, t_Top, 100) 
        loss_boundary = loss_bottom + loss_right + loss_left + loss_front + loss_back + loss_top



        # 连续性损失
        loss_CuCrZr_Cu =Continuity_Loss(model1,model2,x_I2,y_I2,z_I2,t_I2)
        loss_Cu_W =Continuity_Loss(model2,model3,x_I3,y_I3,z_I3,t_I3)
        loss_Q1 = continues_Q(model1,model2,x_I2,y_I2,z_I2,t_I2,k4,k3,7.5)
        loss_Q2 = continues_Q(model2,model3,x_I3,y_I3,z_I3,t_I3,k3,k2,10.5)
        loss_continuity = loss_CuCrZr_Cu + loss_Cu_W + loss_Q2 + loss_Q1


        # 真实值损失(加入少量的数据驱动)
        loss_anaCuCrZr, _ = analytical_loss(model1,Ax_CuCrZr,Ay_CuCrZr,Az_CuCrZr,At_CuCrZr,T_CuCrZr,device)
        loss_anaCu, _ = analytical_loss(model2,Ax_Cu,Ay_Cu,Az_Cu,At_Cu,T_Cu,device)
        loss_anaW, _ = analytical_loss(model3,Ax_W,Ay_W,Az_W,At_W,T_W,device)
        loss_anaW2, _ = analytical_loss(model3,Ax_W2.unsqueeze(-1),Ay_W2.unsqueeze(-1),Az_W2.unsqueeze(-1),At_W2.unsqueeze(-1),T_W2.unsqueeze(-1),device)
        loss_ana = loss_anaCu + loss_anaCuCrZr + loss_anaW 


        # 对流换热
        loss_water,_ = water_boundary_loss(model1, x_I1, y_I1, z_I1, t_I1, 22, k, h)


        # 初始损失
        loss_initCuCrZr= init_loss(model1,Ix_CuCrZr,Iy_CuCrZr,Iz_CuCrZr,30)
        loss_initCu = init_loss(model2, Ix_Cu,Iy_Cu,Iz_Cu,30)
        loss_initW = init_loss(model3,Ix_W,Iy_W,Iz_W,30)
        loss_init = loss_initCuCrZr + loss_initCu + loss_initW

        # ------------------------------------------------------损失更新以及打印损失----------------------------------
        loss = loss_water + 2*loss_continuity + loss_boundary + loss_PDE  + loss_init  + 10*loss_ana

        if i % 100 == 0 :
            print("--------------------------------")
            print(f"当前学习率: {scheduler1.get_last_lr()[0]:.6f}")
            print(f'CuCrZr:{T1.min()},{T1.max()}')
            print(f'Cu:{T2.min()},{T2.max()}')
            print(f'W:{T3.min()},{T3.max()}')
            print(f"loss_water:{loss_water.item()}")
            print(f"loss_continuity:{loss_continuity.item()}")
            print(f"loss_PDE:{loss_PDE.item()}")
            print(f"loss_boundary:{loss_boundary.item()}")
            print(f"loss_ana_all:{loss_ana.item()}")
            print(f"loss_init:{loss_init.item()}")
            print(f"loss:{loss.item()}")
            print("--------------------------------")

        loss_track.append(loss.item())
        pbar.set_description(f"Loss: {loss.item():.15f}")
        loss.backward()
        gradient_list_temp.append(torch.cat([(p.grad.view(-1)) if p.grad is not None else torch.zeros(1).cuda() for p in
                                            model1.parameters()]).cpu().numpy())  # hook gradients from computation graph
        gradient_list_temp1.append(torch.cat([(p.grad.view(-1)) if p.grad is not None else torch.zeros(1).cuda() for p in
                                            model2.parameters()]).cpu().numpy())  # hook gradients from computation graph
        gradient_list_temp2.append(torch.cat([(p.grad.view(-1)) if p.grad is not None else torch.zeros(1).cuda() for p in
                                            model3.parameters()]).cpu().numpy())  # hook gradients from computation graph


        # 使用config用于处理损失冲突的情况
        grad1 = get_gradient_vector(model1)
        grad2 = get_gradient_vector(model2)
        grad3 = get_gradient_vector(model3)
        grads = [grad1,grad2,grad3]
        
        # 更新梯度
        updated_grads = ConFIG_update(grads)
        apply_gradient_vector(model1, updated_grads)
        apply_gradient_vector(model2, updated_grads)
        apply_gradient_vector(model3, updated_grads)

        optim1.step()
        optim2.step()
        optim3.step()

        scheduler1.step()
        scheduler2.step()
        scheduler3.step()

        gradient_list_overall.append(np.mean(np.array(gradient_list_temp), axis=0))
        gradient_list_overall = gradient_list_overall[-10:]
        gradient_list = np.array(gradient_list_overall)
        gradient_variance = (
                np.std(gradient_list, axis=0) / (np.mean(np.abs(gradient_list), axis=0) + 1e-6)).mean()
        gradient_list_temp = []
        if gradient_variance == 0:
            gradient_variance = 1  # for numerical stability

        gradient_list_overall1.append(np.mean(np.array(gradient_list_temp1), axis=0))
        gradient_list_overall1 = gradient_list_overall1[-10:]
        gradient_list1 = np.array(gradient_list_overall1)
        gradient_variance1 = (
                np.std(gradient_list1, axis=0) / (np.mean(np.abs(gradient_list1), axis=0) + 1e-6)).mean()
        gradient_list_temp1 = []
        if gradient_variance1 == 0:
            gradient_variance1 = 1  # for numerical stability

        gradient_list_overall2.append(np.mean(np.array(gradient_list_temp2), axis=0))
        gradient_list_overall2 = gradient_list_overall2[-10:]
        gradient_list2 = np.array(gradient_list_overall2)
        gradient_variance2 = (
                np.std(gradient_list2, axis=0) / (np.mean(np.abs(gradient_list2), axis=0) + 1e-6)).mean()
        gradient_list_temp2 = []
        if gradient_variance2 == 0:
            gradient_variance2 = 1  # for numerical stability

        # 保存最佳训练损失模型：
        if  i % 50 == 0:
            checkpoint_path = f'checkpoints/train_model_{args.model}.pth'
            torch.save({
                'epoch': i,
                'model1_state_dict': model1.state_dict(),
                'model2_state_dict': model2.state_dict(),
                'model3_state_dict': model3.state_dict(),
                'optimizer1_state_dict': optim1.state_dict(),
                'optimizer2_state_dict': optim2.state_dict(),
                'optimizer3_state_dict': optim3.state_dict(),
                'scheduler1_state_dict': scheduler1.state_dict(),
                'scheduler2_state_dict': scheduler2.state_dict(),
                'scheduler3_state_dict': scheduler3.state_dict(),
                'loss': loss.item(),
                'loss_track': loss_track,
            }, checkpoint_path)
            #predict_temperature(checkpoint_path, device, args)

            mae = get_MAE(checkpoint_path, device, args)
            print(f'mae:{mae},best_MAE:{best_MAE}')
            if mae < best_MAE:
                best_MAE = mae
                torch.save({
                    'epoch': i,
                    'model1_state_dict': model1.state_dict(),
                    'model2_state_dict': model2.state_dict(),
                    'model3_state_dict': model3.state_dict(),
                    'optimizer1_state_dict': optim1.state_dict(),
                    'optimizer2_state_dict': optim2.state_dict(),
                    'optimizer3_state_dict': optim3.state_dict(),
                    'scheduler1_state_dict': scheduler1.state_dict(),
                    'scheduler2_state_dict': scheduler2.state_dict(),
                    'scheduler3_state_dict': scheduler3.state_dict(),
                    'loss': loss.item(),
                    'loss_track': loss_track,
                }, f'checkpoints/best_MAE_{args.model}.pth')


        # 重新采样
        if i % 1000 == 0 and i != 0:
            print("重新采样")
            sample(i/10)

if __name__ == "__main__":
    train()