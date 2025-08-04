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
from utils.util import get_n_params,init_weights,make_time_sequence

from model_dict import get_model
from visible2 import predict_temperature,get_MAE
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

train = f'checkpoints/train_model_{args.model}.pth'
MAE = f'checkpoints/best_MAE_{args.model}.pth'
checkpoint_path = MAE
mae = get_MAE(checkpoint_path, device, args)
print(mae)

