import numpy as np
import torch
import torch.nn as nn
import numpy as np
import torch.nn as nn
import copy
#-----------------------------------------FLS------------------------------------------------#
class SinAct(nn.Module):
    def __init__(self):
        super(SinAct, self).__init__()

    def forward(self, x):
        return torch.sin(x)


class Model(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layer):
        super(Model, self).__init__()

        layers = []
        for i in range(num_layer - 1):
            if i == 0:
                layers.append(nn.Linear(in_features=in_dim, out_features=hidden_dim))
                layers.append(SinAct())
            else:
                layers.append(nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
                layers.append(nn.Tanh())

        layers.append(nn.Linear(in_features=hidden_dim, out_features=out_dim))

        self.linear = nn.Sequential(*layers)

    def forward(self, x, y, z, t):
        x = 2*(x - 23.0) / (51.0 - 23.0)-1
        y = 2*(y + 1193.0) / (-1163.0 + 1193.0)-1
        z = 2*(z +12 ) / (0 + 12)-1
        t = 2*(t - 1) / (10 -1 )-1
        #input_tensor = torch.stack([x, y, z, t], dim=-1)
        ini_shape = x.shape
        src = torch.cat((x, y, z, t), dim=-1)
        src = self.linear(src)
        return src.view(ini_shape)