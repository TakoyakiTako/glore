import torch
import torch.nn as nn


class GloRe(nn.Module):
    def __init__(self,channel):
        super(GloRe, self).__init__()
        # node num      : int(channel/4))
        # node channel  : int(channel)
        self.reduce_dim = nn.Conv2d(channel,int(channel), kernel_size=1, stride=1, bias=False)
        self.create_projection_matrix = nn.Conv2d(channel,int(channel/4), kernel_size=1, stride=1, bias=False)
        self.GCN_step_1 = nn.Conv1d(in_channels=int(channel),out_channels=int(channel),kernel_size=1, stride=1,padding=0)
        self.GCN_step_2 = nn.Conv1d(in_channels=int(channel/4),out_channels=int(channel/4),kernel_size=1, stride=1,padding=0)
        self.expand_dim = nn.Conv2d(int(channel),channel, kernel_size=1, stride=1, bias=False)    

    def forward(self, x):
        x_ = self.reduce_dim(x)
        B = self.create_projection_matrix(x)
        # reduce dim
        b,c,h,w = x_.size()
        x_ = x_.view(b,c,h*w)
        # projection matrix
        b,c,h,w = B.size()
        B = B.view(b,c,h*w)
        # b,N,L -> b,L,N
        B = torch.transpose(B, 2, 1)
        # coordinate space -> latent relation space
        V=torch.matmul(x_,B)
        # GCN_1-1
        V1 = self.GCN_step_1(V)
        V1 = torch.transpose(V1, 2, 1)
        # GCN1-2
        V2 = self.GCN_step_2(V1)
        V2 = torch.transpose(V2, 2, 1)
        # Reverse Projection Matrix
        B = torch.transpose(B, 2, 1)
        # latent relation space -> coordinate space
        Y=torch.matmul(V2,B)
        b,c,_ = Y.size()
        # b,c,numpix -> b,c,h,w
        Y = Y.view(b,c,h,w)
        #self.expand_dim
        Y = self.expand_dim(Y)
        
        x = x + Y
        return x