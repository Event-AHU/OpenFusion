# transformer 使用
import numpy as np
import torch.nn as nn
import copy
import numpy as np
import torch
import torch.nn as nn

def get_data(x_range, y_range, x_num, y_num):
    x = np.linspace(x_range[0], x_range[1], x_num)
    t = np.linspace(y_range[0], y_range[1], y_num)

    x_mesh, t_mesh = np.meshgrid(x,t)
    data = np.concatenate((np.expand_dims(x_mesh, -1), np.expand_dims(t_mesh, -1)), axis=-1)
    
    b_left = data[0,:,:] 
    b_right = data[-1,:,:]
    b_upper = data[:,-1,:]
    b_lower = data[:,0,:]
    res = data.reshape(-1,2)

    return res, b_left, b_right, b_upper, b_lower


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def make_time_sequence(src, num_step=5, step=1e-4):
    dim = num_step
    src = np.repeat(np.expand_dims(src, axis=1), dim, axis=1)  # (N, L, 2)
    for i in range(num_step):
        src[:,i,-1] += step*i
    return src


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_data_3d(x_range, y_range, t_range, x_num, y_num, t_num):
    step_x = (x_range[1] - x_range[0]) / float(x_num-1)
    step_y = (y_range[1] - y_range[0]) / float(y_num-1)
    step_t = (t_range[1] - t_range[0]) / float(t_num-1)

    x_mesh, y_mesh, t_mesh = np.mgrid[x_range[0]:x_range[1]+step_x:step_x,y_range[0]:y_range[1]+step_y:step_y,t_range[0]:t_range[1]+step_t:step_t]

    data = np.concatenate((np.expand_dims(x_mesh, -1), np.expand_dims(y_mesh, -1), np.expand_dims(t_mesh, -1)), axis=-1)
    res = data.reshape(-1,3)

    x_mesh, y_mesh, t_mesh = np.mgrid[x_range[0]:x_range[0]+step_x:step_x,y_range[0]:y_range[1]+step_y:step_y,t_range[0]:t_range[1]+step_t:step_t]
    b_left = np.squeeze(np.concatenate((np.expand_dims(x_mesh, -1), np.expand_dims(y_mesh, -1), np.expand_dims(t_mesh, -1)), axis=-1))[1:-1].reshape(-1,3)

    x_mesh, y_mesh, t_mesh = np.mgrid[x_range[1]:x_range[1]+step_x:step_x,y_range[0]:y_range[1]+step_y:step_y,t_range[0]:t_range[1]+step_t:step_t]
    b_right = np.squeeze(np.concatenate((np.expand_dims(x_mesh, -1), np.expand_dims(y_mesh, -1), np.expand_dims(t_mesh, -1)), axis=-1))[1:-1].reshape(-1,3)

    x_mesh, y_mesh, t_mesh = np.mgrid[x_range[0]:x_range[1]+step_x:step_x,y_range[0]:y_range[0]+step_y:step_y,t_range[0]:t_range[1]+step_t:step_t]
    b_lower = np.squeeze(np.concatenate((np.expand_dims(x_mesh, -1), np.expand_dims(y_mesh, -1), np.expand_dims(t_mesh, -1)), axis=-1))[1:-1].reshape(-1,3)

    x_mesh, y_mesh, t_mesh = np.mgrid[x_range[0]:x_range[1]+step_x:step_x,y_range[1]:y_range[1]+step_y:step_y,t_range[0]:t_range[1]+step_t:step_t]
    b_upper = np.squeeze(np.concatenate((np.expand_dims(x_mesh, -1), np.expand_dims(y_mesh, -1), np.expand_dims(t_mesh, -1)), axis=-1))[1:-1].reshape(-1,3)

    return res, b_left, b_right, b_upper, b_lower


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


def rengion_sample(x,y,z,t,sample_num,gradient_variance):
    initial_region = 1e-4
    x_res_region_sample_list = []
    y_res_region_sample_list = []
    z_res_region_sample_list = []
    t_res_region_sample_list = []
    for i in range(sample_num):
        


        x_region_sample = (torch.rand(x.shape).to(x.device)) * np.clip(initial_region / gradient_variance,
                                                                                a_min=0,
                                                                                a_max=0.01)
        y_region_sample = (torch.rand(y.shape).to(y.device)) * np.clip(initial_region / gradient_variance,
                                                                                a_min=0,
                                                                                a_max=0.01)
        z_region_sample = (torch.rand(z.shape).to(z.device)) * np.clip(initial_region / gradient_variance,
                                                                                a_min=0,
                                                                                a_max=0.01)

        t_region_sample = (torch.rand(t.shape).to(t.device)) * np.clip(initial_region / gradient_variance,
                                                                                a_min=0,
                                                                                a_max=0.01)
        x_res_region_sample_list.append(x + x_region_sample)
        y_res_region_sample_list.append(y + y_region_sample)
        z_res_region_sample_list.append(z + z_region_sample)
        t_res_region_sample_list.append(t + t_region_sample)
    x_res_region_sample = torch.cat(x_res_region_sample_list, dim=0)
    y_res_region_sample = torch.cat(y_res_region_sample_list, dim=0)
    z_res_region_sample = torch.cat(z_res_region_sample_list, dim=0)
    t_res_region_sample = torch.cat(t_res_region_sample_list, dim=0)
    return x_res_region_sample,y_res_region_sample,z_res_region_sample,t_res_region_sample
