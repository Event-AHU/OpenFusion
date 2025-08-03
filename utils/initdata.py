import numpy as np
import os
import torch
def make_time_sequence(src, num_step=5, step=1e-4):
    dim = num_step
    src = np.repeat(np.expand_dims(src, axis=1), dim, axis=1)  # (N, L, 2)
    for i in range(num_step):
        src[:,i,-1] += step*i
    return src
def get_analytical_solutions():
    data = np.load("data/data_analysis.npy", allow_pickle=True).item()
    W = data['W']
    Cu = data['Cu']
    CuCrZr =data['CuCrZr']
    return (W,Cu,CuCrZr)

W,Cu,CuCrZr = get_analytical_solutions()

def deal_dataA(domain):
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
    total_points = len(x)
    indices = np.random.choice(total_points, size=2000, replace=False)
    mask = (np.abs(t - 1)<1e-5).squeeze()
    indices = np.where(mask)[0]
    x = x[indices]
    y = y[indices]
    z = z[indices]
    t = t[indices]
    T = T[indices]
    total_points = len(x)
    indices = np.random.choice(total_points, size=2000, replace=False)
    x = x[indices]
    y = y[indices]
    z = z[indices]
    t = t[indices]
    T = T[indices]

    return x,y,z,t,T

def sample_A():
    fact = 1000
    Ax_CuCrZr,Ay_CuCrZr,Az_CuCrZr,At_CuCrZr,T_CuCrZr = deal_dataA(CuCrZr)
    Ax_Cu,Ay_Cu,Az_Cu,At_Cu,T_Cu = deal_dataA(Cu)
    Ax_W,Ay_W,Az_W,At_W,T_W = deal_dataA(W)

    data_to_save = {
        'CuCrZr': {
            'x':Ax_CuCrZr*fact,
            'y': Ay_CuCrZr*fact,
            'z': Az_CuCrZr*fact,
            't': At_CuCrZr,
            'temperature': T_CuCrZr-273.15
        },
        'Cu': {
            'x': Ax_Cu*fact,
            'y': Ay_Cu*fact,
            'z': Az_Cu*fact,
            't': At_Cu,
            'temperature': T_Cu-273.15
        },
        'W': {
            'x': Ax_W*fact,
            'y': Ay_W*fact,
            'z': Az_W*fact,
            't': At_W,
            'temperature': T_W-273.15
        }
    }
    print(At_W)
    print(len(At_W))
    print(T_W.min())
    np.save('data/init_data.npy', data_to_save)
sample_A()