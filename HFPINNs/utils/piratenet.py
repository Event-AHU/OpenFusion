import torch
import torch.nn as nn
import math

class PeriodEmbs(nn.Module):
    def __init__(self, period, axis, trainable):
        super().__init__()
        self.axis = axis
        self.trainable = trainable
        self.period_params = nn.ParameterDict()
        
        for idx, is_trainable in enumerate(trainable):
            if is_trainable:
                self.period_params[f"period_{idx}"] = nn.Parameter(torch.tensor(period[idx]))
            else:
                self.register_buffer(f"period_{idx}", torch.tensor(period[idx]))
                
    def forward(self, x):
        y = []
        for i in range(x.shape[1]):
            if i in self.axis:
                idx = self.axis.index(i)
                period = self.period_params[f"period_{idx}"] if self.trainable[idx] else getattr(self, f"period_{idx}")
                y.extend([torch.cos(period * x[:, i]), torch.sin(period * x[:, i])])
            else:
                y.append(x[:, i])
        return torch.stack(y, dim=1)

class FourierEmbs(nn.Module):
    def __init__(self, embed_scale, embed_dim):
        super().__init__()
        self.embed_scale = embed_scale
        self.embed_dim = embed_dim
        self.kernel = nn.Parameter(torch.randn(4, embed_dim // 2) * embed_scale)
        
    def forward(self, x):
        proj = torch.matmul(x, self.kernel)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)

class Embedding(nn.Module):
    def __init__(self, periodicity=None, fourier_emb=None):
        super().__init__()
        self.periodicity = periodicity
        self.fourier_emb = fourier_emb
        
        if periodicity:
            self.period_emb = PeriodEmbs(**periodicity)
        if fourier_emb:
            self.fourier = FourierEmbs(**fourier_emb)
            
    def forward(self, x):
        if self.periodicity:
            x = self.period_emb(x)
        if self.fourier_emb:
            x = self.fourier(x)
        return x

class PIModifiedBottleneck(nn.Module):
    def __init__(self, hidden_dim, output_dim, activation='tanh', nonlinearity=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.nonlinearity = nonlinearity
        
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Activation {activation} not supported")
            
        self.fc1 = nn.Linear(output_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim) 
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.alpha = nn.Parameter(torch.tensor(nonlinearity))
        
    def forward(self, x, u, v):
        identity = x
        
        x = self.fc1(x)
        x = self.activation(x)
        x = x * u + (1 - x) * v
        
        x = self.fc2(x)
        x = self.activation(x)
        x = x * u + (1 - x) * v
        
        x = self.fc3(x)
        x = self.activation(x)
        
        x = self.alpha * x + (1 - self.alpha) * identity
        return x

class Model(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=256, output_dim=1, num_layers=2, 
                 activation='tanh', nonlinearity=0.0, periodicity=None, fourier_emb=None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.activation = activation
        self.nonlinearity = nonlinearity
        
        self.embedding = Embedding(periodicity, fourier_emb)
        
        if activation == 'tanh':
            self.activation_fn = nn.Tanh()
        elif activation == 'relu':
            self.activation_fn = nn.ReLU()
        else:
            raise ValueError(f"Activation {activation} not supported")
            
        embed_dim = fourier_emb['embed_dim'] if fourier_emb else input_dim
        self.u_net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            self.activation_fn
        )
        
        self.v_net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            self.activation_fn
        )
        
        self.bottleneck_layers = nn.ModuleList([
            PIModifiedBottleneck(hidden_dim, embed_dim, activation, nonlinearity)
            for _ in range(num_layers)
        ])
        
        self.output_layer = nn.Linear(embed_dim, output_dim)
        
    def forward(self, x,y,z,t):
        x = 2 * (x - 23.0) / (51.0 - 23.0) - 1
        y = 2 * (y + 1193.0) / (-1163.0 + 1193.0) - 1
        z = 2 * (z + 12) / (0 + 12) - 1
        t = 2 * (t - 1) / (10 - 1) - 1
        x = torch.cat([x,y,z,t],dim=-1)
        
        x = self.embedding(x)
        
        u = self.u_net(x)
        v = self.v_net(x)
        
        for layer in self.bottleneck_layers:
            x = layer(x, u, v)
            
        y = self.output_layer(x)
        return y 