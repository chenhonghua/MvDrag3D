import functools
import math
import os
import time
from tkinter import W

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import torch.nn.init as init


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, input_dims, i=0):
    if i == -1:
        return nn.Identity(), input_dims
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : input_dims,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


class Linear_Res(nn.Module):
    def __init__(self, W):
        super(Linear_Res, self).__init__()
        self.main_stream = nn.Linear(W, W)

    def forward(self, x):
        x = F.relu(x)
        return x + self.main_stream(x)


class Head_Res_Net(nn.Module):
    def __init__(self, W, H):
        super(Head_Res_Net, self).__init__()
        self.W = W
        self.H = H

        self.feature_out = [Linear_Res(self.W)]
        self.feature_out.append(nn.Linear(W, self.H))
        self.feature_out = nn.Sequential(*self.feature_out)
    
    def initialize_weights(self,):
        for m in self.feature_out.modules():
            if isinstance(m, nn.Linear):
                init.constant_(m.weight, 0)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        return self.feature_out(x)


class MLP_Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP_Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# deformation network for deforming four gaussian blobs: xyz and shs
class Deformation(nn.Module):
    def __init__(self):
        super(Deformation, self).__init__()

        hidden_dim = 256  # Example hidden dimension

        self.pe, _ = get_embedder(multires=6, input_dims=39) #3-21 10-63 5-33 8-51 6-39 7-45

        self.pos_deform_gs0 = Head_Res_Net(39, 3)
        self.pos_deform_gs1 = Head_Res_Net(39, 3)
        self.pos_deform_gs2 = Head_Res_Net(39, 3)
        self.pos_deform_gs3 = Head_Res_Net(39, 3)


    def forward(self, gs0_pts, gs1_pts, gs2_pts, gs3_pts):
        dx0 = self.pos_deform_gs0(self.pe(gs0_pts))
        dx1 = self.pos_deform_gs1(self.pe(gs1_pts))
        dx2 = self.pos_deform_gs2(self.pe(gs2_pts))
        dx3 = self.pos_deform_gs3(self.pe(gs3_pts))

        return dx0, dx1, dx2, dx3
    
    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if "grid" not in name:
                parameter_list.append(param)
        return parameter_list


class deform_network(nn.Module):
    def __init__(self):
        super(deform_network, self).__init__()
        posbase_pe= 10
        self.use_res = True
        if self.use_res:
            print("Using zero-init and residual")
        self.deformation_net = Deformation()
        self.register_buffer('pos_poc', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
        self.apply(initialize_weights)

        if self.use_res:
            self.deformation_net.pos_deform_gs0.initialize_weights()
            self.deformation_net.pos_deform_gs1.initialize_weights()
            self.deformation_net.pos_deform_gs2.initialize_weights()
            self.deformation_net.pos_deform_gs3.initialize_weights()

    def forward(self, gs0_pts, gs1_pts, gs2_pts, gs3_pts):
        dx0, dx1, dx2, dx3= self.deformation_net(gs0_pts, gs1_pts, gs2_pts, gs3_pts)
        return dx0, dx1, dx2, dx3
    
    def get_mlp_parameters(self):
        return self.deformation_net.get_mlp_parameters()
   

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight,gain=1)
        if m.bias is not None:
            init.xavier_uniform_(m.weight,gain=1)

def initialize_zeros_weights(m):
    if isinstance(m, nn.Linear):
        init.constant_(m.weight, 0)
        if m.bias is not None:
            init.constant_(m.bias, 0)


if __name__ == '__main__':
    # net = deform_network()
    # print(net)

    # Create an embedder with specific parameters
    embed, out_dim = get_embedder(multires=3, input_dims=3)

    # Example input tensor
    inputs = torch.randn(5, 3)  # Batch of 5 inputs, each of dimension 3

    # Apply the embedding
    embedded_inputs = embed(inputs)
    print(embedded_inputs.shape)  # Should print (5, out_dim)
