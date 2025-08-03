import numpy as np
import torch
import torch.nn as nn
import numpy as np
import torch.nn as nn
import copy
#----------------------------------------------------------QRes-----------------------------------------------#
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class QRes_block(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(QRes_block, self).__init__()
        self.H1 = nn.Linear(in_features=in_dim, out_features=out_dim)
        self.H2 = nn.Linear(in_features=in_dim, out_features=out_dim)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x1 = self.H1(x)
        x2 = self.H2(x)
        return self.act(x1 * x2 + x1)

class Model(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layer):
        super(Model, self).__init__()
        self.N = num_layer - 1
        self.inlayer = QRes_block(in_dim, hidden_dim)
        self.layers = get_clones(QRes_block(hidden_dim, hidden_dim), num_layer - 1)
        self.outlayer = nn.Linear(in_features=hidden_dim, out_features=out_dim)

    def forward(self, x, y, z, t):
        x = 2*(x - 23.0) / (51.0 - 23.0)-1
        y = 2*(y + 1193.0) / (-1163.0 + 1193.0)-1
        z = 2*(z +12 ) / (0 + 12)-1
        t = 2*(t - 1) / (10 -1 )-1
        #input_tensor = torch.stack([x, y, z, t], dim=-1)
        ini_shape = x.shape
        src = torch.cat((x, y, z, t), dim=-1)
        src = self.inlayer(src)
        for i in range(self.N):
            src = self.layers[i](src)
        src = self.outlayer(src)
        return src.view(ini_shape)