import torch
import math

def neumann_boundary_loss(network, x, y, z, t, coordinate):   
    T = network(x,y,z,t)
    T_x = torch.autograd.grad(T, x, grad_outputs=torch.ones_like(T), retain_graph=True, create_graph=True)[0]
    T_y = torch.autograd.grad(T, y, grad_outputs=torch.ones_like(T), retain_graph=True, create_graph=True)[0]
    T_z = torch.autograd.grad(T, z, grad_outputs=torch.ones_like(T), retain_graph=True, create_graph=True)[0]
    if coordinate == "X":
        residual = T_x.pow(2)
    elif coordinate == "Y":
        residual = T_y.pow(2)
    elif coordinate == "Z":
        residual = T_z.pow(2)
    return residual.mean()


def water_boundary_loss(network, x, y, z, t, temperature_water, k, h):
    T = network(x, y, z, t)
    center_x = 36.920
    center_y = -1179.510
    radius = 6
    r = torch.sqrt(((x-center_x)**2 + (y-center_y)**2))
    if not torch.allclose(r, torch.full_like(r, radius), atol=1e-3):
        raise ValueError("r=6")

    vec_x = x - center_x 
    vec_y = y - center_y

    normal_x = vec_x / radius  
    normal_y = vec_y / radius  
    normal_z = torch.zeros_like(x)  

    T_x = torch.autograd.grad(T, x, grad_outputs=torch.ones_like(T), retain_graph=True, create_graph=True)[0]
    T_y = torch.autograd.grad(T, y, grad_outputs=torch.ones_like(T), retain_graph=True, create_graph=True)[0]
    T_z = torch.autograd.grad(T, z, grad_outputs=torch.ones_like(T), retain_graph=True, create_graph=True)[0]

    dT_dn = T_x * normal_x + T_y * normal_y + T_z * normal_z
    residual = (-k *dT_dn - h * (T - temperature_water))/h
    return (residual).pow(2).mean(),T


def constant_temperature_boundary_loss(network, x, y, z, t, value_b): 
    x_temp =x /1000 -0.023
    Temp=300*torch.exp(-(x_temp-0.014)**2/(0.006*0.006))
    boundary_loss=torch.mean((network(x,y,z,t)-Temp)**2)
    return boundary_loss


def Continuity_Loss(network1,network2,x,y,z,t):
    T1 = network1(x,y,z,t)
    T2 = network2(x,y,z,t)
    residual = T1-T2
    return residual.pow(2).mean()


def pde_loss(network, x, y, z, t, k, rho, cp):
    T = network(x, y, z, t)
    T_x = torch.autograd.grad(T, x, grad_outputs=torch.ones_like(T), retain_graph=True, create_graph=True)[0]
    T_y = torch.autograd.grad(T, y, grad_outputs=torch.ones_like(T), retain_graph=True, create_graph=True)[0]
    T_z = torch.autograd.grad(T, z, grad_outputs=torch.ones_like(T), retain_graph=True, create_graph=True)[0]
    T_t = torch.autograd.grad(T, t, grad_outputs=torch.ones_like(T), retain_graph=True, create_graph=True)[0]
    kT_x = k * T_x
    kT_y = k * T_y
    kT_z = k * T_z   
    kT_xx = torch.autograd.grad(kT_x, x, grad_outputs=torch.ones_like(T_x), retain_graph=True, create_graph=True)[0]
    kT_yy = torch.autograd.grad(kT_y, y, grad_outputs=torch.ones_like(T_y), retain_graph=True, create_graph=True)[0]
    kT_zz = torch.autograd.grad(kT_z, z, grad_outputs=torch.ones_like(T_z), retain_graph=True, create_graph=True)[0]
    residual = ((kT_xx + kT_yy + kT_zz) - rho * cp * T_t)/(rho * cp)
    return residual.pow(2).mean(),T


def analytical_loss(model, x, y, z, t, gt, device):
    x = torch.tensor(x, dtype=torch.float32, requires_grad=True).to(device)
    y = torch.tensor(y, dtype=torch.float32, requires_grad=True).to(device)
    z = torch.tensor(z, dtype=torch.float32, requires_grad=True).to(device)
    t = torch.tensor(t, dtype=torch.float32, requires_grad=True).to(device)
    gt = torch.tensor(gt, dtype=torch.float32, requires_grad=True).to(device)
    T = model(x, y, z, t)
    residual = T - gt
    loss = residual.pow(2).mean()
    return loss,T

def init_loss(model, x, y, z, init_temp):
    t = torch.full_like(x, fill_value=0)  
    T = model(x, y, z, t)
    residual = T -init_temp
    return residual.pow(2).mean()
    

def continues_Q(network1,network2,x,y,z,t,k1,k2,radius):
    T1 = network1(x, y, z, t)
    T2 = network2(x, y, z, t)
    center_x = 36.920
    center_y = -1179.510
    r = torch.sqrt(((x-center_x)**2 + (y-center_y)**2))
    if not torch.allclose(r, torch.full_like(r, radius), atol=1e-3):
        raise ValueError("r=6")
    vec_x = x - center_x 
    vec_y = y - center_y
    normal_x = vec_x / r
    normal_y = vec_y / r
    normal_z = torch.zeros_like(x)  
    
    T_x = torch.autograd.grad(T1, x, grad_outputs=torch.ones_like(T1), retain_graph=True, create_graph=True)[0]
    T_y = torch.autograd.grad(T1, y, grad_outputs=torch.ones_like(T1), retain_graph=True, create_graph=True)[0]
    T_z = torch.autograd.grad(T1, z, grad_outputs=torch.ones_like(T1), retain_graph=True, create_graph=True)[0]

    T_x2 = torch.autograd.grad(T2, x, grad_outputs=torch.ones_like(T2), retain_graph=True, create_graph=True)[0]
    T_y2 = torch.autograd.grad(T2, y, grad_outputs=torch.ones_like(T2), retain_graph=True, create_graph=True)[0]
    T_z2 = torch.autograd.grad(T2, z, grad_outputs=torch.ones_like(T2), retain_graph=True, create_graph=True)[0]
    #breakpoint()
    dT_dn1 = T_x * normal_x + T_y * normal_y + T_z * normal_z
    dT_dn2 = T_x2 * normal_x + T_y2 * normal_y + T_z2 * normal_z

    residual = k1*dT_dn1 - k2*dT_dn2
    return (residual).pow(2).mean()