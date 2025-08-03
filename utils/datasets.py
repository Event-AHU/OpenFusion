import numpy as np
import os
import torch
from typing import List

def make_time_sequence(src, num_step=5, step=1e-4):
    dim = num_step
    src = np.repeat(np.expand_dims(src, axis=1), dim, axis=1)  # (N, L, 2)
    for i in range(num_step):
        src[:,i,-1] += step*i
    return src

class DataProcessor:
    def __init__(self):
        pass

    def load_data(self, filename):
        data = np.load(filename, allow_pickle=True)
        return data
    
    def read_specific_lines(self, filename):
        with open(filename, 'r', encoding='ISO-8859-1') as file:
            lines = file.readlines()
            data = np.array([line.strip().split() for line in lines[1:]])
            x = data[:, 1].astype(float)  
            y = data[:, 2].astype(float)  
            z = data[:, 3].astype(float)  
            temperature = data[:, 4].astype(float)  
        return x, y, z, temperature

    def process_data(self, input_file,output_dir, name, i):
        file_path = os.path.join(output_dir, name)
        x, y, z, temperature = self.read_specific_lines(input_file)
        t = np.full(x.shape, i)
        np.save(file_path, {
            'x': x,
            'y': y,
            'z': z,
            't': t,
            'temperature': temperature
        })
    
    def is_in_ring(self,points, center, inner_radius, outer_radius):
        x = points['x'] 
        z = points['z']
        distances = np.sqrt((x - center[0])**2 + (z - center[1])**2)
        
        return (distances >= inner_radius) & (distances <= outer_radius)
    
    def merge_data(self,data_path,name_list):
        data_list = []
        for i in range(0,10):
            file_path = os.path.join(data_path, name_list[i])
            data = data = np.load(file_path, allow_pickle=True).item()
            data_list.append(data)

        merged_data = {}
        for key in data_list[0].keys():
            merged_data[key] = []
        for data in data_list:
            for key in data.keys():
                if isinstance(data[key], np.ndarray):
                    merged_data[key].append(data[key])
                else:
                    merged_data[key].append(data[key])
        for key in merged_data.keys():
            merged_data[key] = np.array(merged_data[key])

        # print(merged_data['x'][0][49632],merged_data['y'][0][49632],merged_data['z'][0][49632],merged_data['t'][0][49632])
        merged_data['x'] = merged_data['x'].reshape(-1, 1)  # [402690, 1]
        merged_data['y'] = merged_data['y'].reshape(-1, 1)  
        merged_data['z'] = merged_data['z'].reshape(-1, 1)  
        merged_data['t'] = merged_data['t'].reshape(-1, 1)  
        merged_data['temperature'] = merged_data['temperature'].reshape(-1, 1)  
        file = f"{output_file}/merged_data.npy"
        np.save(file, merged_data)
        return file

    def get_domain(self,data_path):
        data = np.load(data_path, allow_pickle=True).item()
        # the centre of a circle
        x = 36.920/1000
        z = -1179.510/1000

        # W
        total_points = len(data['x'])
        inner_radius = 10.5/1000
        outer_radius = 9999/1000
        max_points = 0
        center = (x, z)
        mask = self.is_in_ring(data, center, inner_radius, outer_radius)
        points_count = np.sum(mask)
        max_points = points_count
        final_mask = self.is_in_ring(data, center, inner_radius, outer_radius)
        W = np.column_stack((data['x'][final_mask] , data['z'][final_mask] , data['y'][final_mask] , 
                        data['t'][final_mask],data['temperature'][final_mask] + 273.15))
        
        # Cu
        total_points = len(data['x'])
        inner_radius = 7.5/1000
        outer_radius = 10.5/1000
        max_points = 0
        center = (x, z)
        mask = self.is_in_ring(data, center, inner_radius, outer_radius)
        points_count = np.sum(mask)
        max_points = points_count
        final_mask = self.is_in_ring(data, center, inner_radius, outer_radius)
        Cu = np.column_stack((data['x'][final_mask] , data['z'][final_mask] , data['y'][final_mask] , 
                        data['t'][final_mask],data['temperature'][final_mask] + 273.15))

        # CuCrZr
        total_points = len(data['x'])
        inner_radius = 6/1000
        outer_radius = 7.5/1000
        max_points = 0
        center = (x, z)
        mask = self.is_in_ring(data, center, inner_radius, outer_radius)
        points_count = np.sum(mask)
        max_points = points_count
        final_mask = self.is_in_ring(data, center, inner_radius, outer_radius)
        CuCrZr = np.column_stack((data['x'][final_mask] , data['z'][final_mask] , data['y'][final_mask] , 
                        data['t'][final_mask],data['temperature'][final_mask] + 273.15))
        
        sampling_data = {
            'W': W,
            'Cu': Cu,
            'CuCrZr': CuCrZr,
        }
        np.save('data/data_analysis.npy', sampling_data)

class get_Dataset():
    def __init__(self,device) -> None:
        self.device = device

    def get_analytical_solutions(self):
        data = np.load("data/Adata.npy", allow_pickle=True).item()
        W = data['W']
        Cu = data['Cu']
        CuCrZr =data['CuCrZr']
        return (W,Cu,CuCrZr)

    def get_analytical_solutions2(self):
        data = np.load("data/Adata2.npy", allow_pickle=True).item()
        W = data['W']
        Cu = data['Cu']
        CuCrZr =data['CuCrZr']
        return (W,Cu,CuCrZr)
    
    def get_init(self):
        data = np.load("data/init_data.npy", allow_pickle=True).item()
        W = data['W']
        Cu = data['Cu']
        CuCrZr =data['CuCrZr']
        return (W,Cu,CuCrZr)

    def get_analytical_solutionsALL(self):
        data = np.load("data/merged_data.npy", allow_pickle=True).item()
        return (data)
    
    def sample_boundary(self):
        data = np.load("data/sampling_points_3d.npy", allow_pickle=True).item()  # 将数组转换为字典
        R_bottom = data['Bottom']
        R_left = data['Left']
        R_right = data['Right']
        R_top = data['Top']
        R_front = data['Front']
        R_back = data['Back']
        return (R_bottom,R_left,R_right,R_top,R_front,R_back)   
    
    def sample_pde(self):
        data = np.load("data/sampling_points_3d.npy", allow_pickle=True).item()  # 将数组转换为字典
        A2 = data['A2']
        A3 = data['A3']
        A4 = data['A4']
        return (A2,A3,A4)
    
    def sample_interface(self):
        data = np.load("data/sampling_points_3d.npy", allow_pickle=True).item()  # 将数组转换为字典
        I1 = data['I1']
        I2 = data['I2']
        I3 = data['I3']
        return (I1,I2,I3)

    def deal_data(self,domain):
        x = domain[:, 0].reshape(-1, 1)  
        y = domain[:, 1].reshape(-1, 1)  
        z = domain[:, 2].reshape(-1, 1)  
        t = domain[:, 3].reshape(-1, 1)  

        x = torch.tensor(x, dtype=torch.float32, requires_grad=True).to(self.device)
        y = torch.tensor(y, dtype=torch.float32, requires_grad=True).to(self.device)
        z = torch.tensor(z, dtype=torch.float32, requires_grad=True).to(self.device)
        t = torch.tensor(t, dtype=torch.float32, requires_grad=True).to(self.device)
        return x,y,z,t

    def deal_data_seq(self,domain,num_step=2, step=1e-2):
        x = domain[:, 0].reshape(-1, 1)  
        y = domain[:, 1].reshape(-1, 1)  
        z = domain[:, 2].reshape(-1, 1)  
        t = domain[:, 3].reshape(-1, 1)  

        x = np.expand_dims(np.tile(x[:], (num_step)), -1)
        y = np.expand_dims(np.tile(y[:], (num_step)), -1)
        z = np.expand_dims(np.tile(z[:], (num_step)), -1)
        t = make_time_sequence(t, num_step, step)

        x = torch.tensor(x, dtype=torch.float32, requires_grad=True).to(self.device)
        y = torch.tensor(y, dtype=torch.float32, requires_grad=True).to(self.device)
        z = torch.tensor(z, dtype=torch.float32, requires_grad=True).to(self.device)
        t = torch.tensor(t, dtype=torch.float32, requires_grad=True).to(self.device)
        return x,y,z,t
    

    def deal_dataA(self, domain):
        if isinstance(domain, dict):
            x = np.array(domain['x'], dtype=np.float32)
            y = np.array(domain['y'], dtype=np.float32)
            z = np.array(domain['z'], dtype=np.float32)
            t = np.array(domain['t'], dtype=np.float32)
            T = np.array(domain['temperature'], dtype=np.float32)
        else:
            # 处理数组类型的数据
            x = domain[:, 0].reshape(-1, 1).astype(np.float32)
            y = domain[:, 1].reshape(-1, 1).astype(np.float32)
            z = domain[:, 2].reshape(-1, 1).astype(np.float32)
            t = domain[:, 3].reshape(-1, 1).astype(np.float32)
            T = domain[:, 4].reshape(-1, 1).astype(np.float32)
            total_points = len(x)

        x = torch.tensor(x, dtype=torch.float32, requires_grad=True).to(self.device)
        y = torch.tensor(y, dtype=torch.float32, requires_grad=True).to(self.device)
        z = torch.tensor(z, dtype=torch.float32, requires_grad=True).to(self.device)
        t = torch.tensor(t, dtype=torch.float32, requires_grad=True).to(self.device)
        T = torch.tensor(T, dtype=torch.float32, requires_grad=True).to(self.device)

        return x,y,z,t,T

if __name__ == "__main__":

    data = DataProcessor()
    output_file = 'data'
    name_list = []
    for i in range(1,11):
        input_file = f'data/temp_varity/Temp300&h29097&{i}s.txt'
        name = f'data_{i}s.npy'
        name_list.append(name)
        get_data =  data.process_data(input_file,output_file, name, i)
    file = data.merge_data(output_file,name_list)
    data.get_domain(file)
    
    data = get_Dataset('cuda:0')
    D1, D2, D3 = data.sample_pde()
    I1, I2, I3 = data.sample_interface()
    Bottom,Left,Right,Top,Front,Back = data.sample_boundary()
    W,Cu,CuCrZr = data.get_analytical_solutions()
    data_all =  data.get_analytical_solutionsALL()
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
    # Analytical solution
    Ax_CuCrZr,Ay_CuCrZr,Az_CuCrZr,At_CuCrZr,T_CuCrZr = data.deal_dataA(CuCrZr)
    Ax_Cu,Ay_Cu,Az_Cu,At_Cu,T_Cu = data.deal_dataA(Cu)
    Ax_W,Ay_W,Az_W,At_W,T_W = data.deal_dataA(W)
    x_all,y_all,z_all,t_all,T_all = data.deal_dataA(data_all)
    # Inital
    Ix_CuCrZr,Iy_CuCrZr,Iz_CuCrZr,It_CuCrZr,IT_CuCrZr = data.deal_dataA(CuCrZr)
    Ix_Cu,Iy_Cu,Iz_Cu,It_Cu,IT_Cu = data.deal_dataA(Cu)
    Ix_W,Iy_W,Iz_W,It_W,IT_W = data.deal_dataA(W)

    print("z_CuCrZr : {:.10f} ~ {:.10f}".format(z_CuCrZr.min().item(), z_CuCrZr.max().item()))
    print("z_Cu : {:.10f} ~ {:.10f}".format(z_Cu.min().item(), z_Cu.max().item()))
    print("z_W : {:.10f} ~ {:.10f}".format(z_W.min().item(), z_W.max().item()))
    
    print("z_I1 : {:.10f} ~ {:.10f}".format(z_I1.min().item(), z_I1.max().item()))
    print("z_I2 : {:.10f} ~ {:.10f}".format(z_I2.min().item(), z_I2.max().item()))
    print("z_I3 : {:.10f} ~ {:.10f}".format(z_I3.min().item(), z_I3.max().item()))
    
    print("z_Left : {:.10f} ~ {:.10f}".format(z_Left.min().item(), z_Left.max().item()))
    print("z_Right : {:.10f} ~ {:.10f}".format(z_Right.min().item(), z_Right.max().item()))
    print("z_Front : {:.10f} ~ {:.10f}".format(z_Front.min().item(), z_Front.max().item()))
    print("z_Back : {:.10f} ~ {:.10f}".format(z_Back.min().item(), z_Back.max().item()))
    print("z_Bottom : {:.10f} ~ {:.10f}".format(z_Bottom.min().item(), z_Bottom.max().item()))
    print("z_Top : {:.10f} ~ {:.10f}".format(z_Top.min().item(), z_Top.max().item()))
    
    print("Az_CuCrZr : {:.10f} ~ {:.10f}".format(Az_CuCrZr.min().item(), Az_CuCrZr.max().item()))
    print("Az_Cu : {:.10f} ~ {:.10f}".format(Az_Cu.min().item(), Az_Cu.max().item()))
    print("Az_W : {:.10f} ~ {:.10f}".format(Az_W.min().item(), Az_W.max().item()))

    print("Az_CuCrZr : {:.10f} ~ {:.10f}".format(Az_CuCrZr.min().item(), Az_CuCrZr.max().item()))
    print("Az_Cu : {:.10f} ~ {:.10f}".format(Az_Cu.min().item(), Az_Cu.max().item()))
    print("Az_W : {:.10f} ~ {:.10f}".format(Az_W.min().item(), Az_W.max().item()))
    print("x_all : {:.10f} ~ {:.10f}".format(x_all.min().item(), x_all.max().item()))


    