import numpy as np
import torch
import torch.nn as nn
import numpy as np
import torch.nn as nn
import copy
#----------------------------------------------------------MLP-----------------------------------------------#
class Swish(nn.Module):
    def __init__(self, trainable=True):
        super().__init__()
    
    def forward(self, x):
        # 如果β不可训练，直接使用固定值
        return x * torch.sigmoid(x)

class Model(nn.Module):
    def __init__(self, in_dim=4, hidden_dim=512, out_dim=1, num_layer=4):
        super(Model, self).__init__()
        
        self.ini_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh()
        )

        layers = []
        for _ in range(num_layer):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        
        self.out_net = nn.Sequential(
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x, y, z, t):
        x = 2*(x - 23.0) / (51.0 - 23.0)-1
        y = 2*(y + 1193.0) / (-1163.0 + 1193.0)-1
        z = 2*(z +12 ) / (0 + 12)-1
        t = 2*(t - 1) / (10 -1 )-1
        #x = 2*(x - self.x_min) / (self.x_max - self.x_min)-1
        #y = (y - self.y_min) / (self.y_max - self.y_min)
        #z = (z - self.z_min) / (self.z_max - self.z_min)
        #t = (t - self.t_min) / (self.t_max - self.t_min)
        input_tensor = torch.stack([x, y, z, t], dim=-1)
        ini_shape = x.shape
        T = self.ini_net(input_tensor)
        T = self.net(T)
        T = self.out_net(T)
        #T = 100 + (y-1) * T.view(ini_shape)
        # return T.view(ini_shape) 
        return T.view(ini_shape)