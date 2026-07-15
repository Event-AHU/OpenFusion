import sys
sys.path.append(".")
import torch
import torch.nn as nn
from model.DiffTransformer.multihead_flashdiff_1 import MultiheadFlashDiff1,MultiheadCrossFlashDiff1
class SpatialBlock(nn.Module):
    def __init__(self,embed_dim=512,num_heads=8,mlp_ratio=4,layers=2):
        super(SpatialBlock,self).__init__()
        self.dim=embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim//num_heads
        assert self.dim == self.num_heads*self.head_dim,"dim必须被num_heads整除"
        self.hidden_dim = embed_dim*mlp_ratio

        self.layers = layers
        self.attn_list = [MultiheadFlashDiff1(embed_dim=self.dim,depth=i,num_heads=self.num_heads) for i in range(self.layers)]
        self.attnblock = torch.nn.ModuleList(self.attn_list)
    
    def forward(self,x):
        B,E,H,W = x.shape
        x = x.view(B,E,-1).permute(0,2,1).contiguous()
        for layer in self.attnblock:
            x = layer(x)
        y = x.permute(0,2,1).contiguous().view(B,E,H,W)
        return y

class Temporal(nn.Module):
    def __init__(self,embed_dim=512,num_heads=8,mlp_ratio=4,frame_size=3):
        super(Temporal,self).__init__()
        self.frame_size=frame_size
        self.dim=embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim//num_heads
        assert self.dim == self.num_heads*self.head_dim,"dim必须被num_heads整除"
        self.hidden_dim = embed_dim*mlp_ratio

        self.layers = 2
        self.attn_list = [MultiheadFlashDiff1(embed_dim=self.dim,depth=i,num_heads=self.num_heads) for i in range(self.layers)]
        self.attnblock = torch.nn.ModuleList(self.attn_list)
    
    def forward(self,x):
        F,E,H,W = x.shape
        B = F//self.frame_size
        x = x.reshape(B,self.frame_size,E,H,W).permute(0,1,3,4,2)
        x = x.reshape(B,self.frame_size,-1,E)
        x,y,z = x[:,0,:,:],x[:,1,:,:],x[:,2,:,:]
        y = self.attn_list[0](x,y)
        z = self.attn_list[1](y,z)
        out = torch.cat([x,y,z],dim=1).reshape(B,self.frame_size,H,W,E).reshape(B*self.frame_size,H,W,E).permute(0,3,1,2)
        return out

class TemporalBlock(nn.Module):
    def __init__(self,embed_dim=512,num_heads=8,mlp_ratio=4,layers=2,frame_size=3):
        super(TemporalBlock,self).__init__()
        self.frame_size=frame_size
        self.dim=embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim//num_heads
        assert self.dim == self.num_heads*self.head_dim,"dim必须被num_heads整除"
        self.hidden_dim = embed_dim*mlp_ratio

        self.layers = layers
        self.T_DiffFormer_list=[Temporal(embed_dim=embed_dim,num_heads=num_heads,frame_size=frame_size) for i in range(layers)]
        self.T_DiffFormer_block = nn.ModuleList(self.T_DiffFormer_list)
    
    def forward(self,x):
        for block in self.T_DiffFormer_block:
            x = block(x)
        return x


if __name__=='__main__':
    x = torch.zeros((6,320,64,64))
    db= TemporalBlock(320,8,layers=3,frame_size=3)
    y = db(x)
    print(y.shape)


