from typing import List, Optional, Union, Any

import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.nn import HGTConv, GCNConv, GAE

def get_activation(act):
        if act == 'relu':
            return nn.ReLU()
        elif act == 'selu':
            return nn.SELU()
        elif act == 'sigmoid':
            return nn.Sigmoid()
        elif act == 'elu':
            return nn.ELU()
        else:
            raise ValueError(f"Activation function {act} is not supported.")
    
def mlp_encoder_(input_dim, dim_list,act='elu'):
    layers = []
    prev_dim = input_dim
    for dim in dim_list[:-1]:
        layers.append(nn.Linear(prev_dim, dim))
        # layers.append(nn.BatchNorm1d(dim, affine=True))
        # layers.append(nn.Dropout(p=self.dropout))
        layers.append(get_activation(act))
        prev_dim = dim
    layers.append(nn.Linear(prev_dim,dim_list[-1]))
    return nn.Sequential(*layers)
    
def mlp_encoder(input_dim, dim_list,act='elu'):
    layers = []
    prev_dim = input_dim
    for dim in dim_list:
        layers.append(nn.Linear(prev_dim, dim))
        # layers.append(nn.BatchNorm1d(dim, affine=True))
        # layers.append(nn.Dropout(p=self.dropout))
        layers.append(get_activation(act))
        prev_dim = dim
    # layers.append(nn.Linear(prev_dim,dim_list[-1]))
    return nn.Sequential(*layers)

def sym_norm(edge_index:torch.Tensor,
             num_nodes:int,
             edge_weight:Optional[Union[Any,torch.Tensor]]=None,
             improved:Optional[bool]=False,
             dtype:Optional[Any]=None
    )-> List:
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype, device=edge_index.device)

    fill_value = 1 if not improved else 2
    edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes)

    row, col = edge_index

    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

class WLGCN_vanilla(MessagePassing):
    def __init__(self, K:Optional[int]=1,
                 cached:Optional[bool]=False,
                 bias:Optional[bool]=True,
                 **kwargs):
        super(WLGCN_vanilla, self).__init__(aggr='add', **kwargs)
        self.K = K
        
    def forward(self, x:torch.Tensor,
                edge_index:torch.Tensor,
                edge_weight:Union[torch.Tensor,None]=None):
        edge_index, norm = sym_norm(edge_index, x.size(0), edge_weight,
                                        dtype=x.dtype)

        xs = [x]
        for k in range(self.K):
            xs.append(self.propagate(edge_index, x=xs[-1], norm=norm))
        return torch.cat(xs, dim = 1)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {}, K={})'.format(self.__class__.__name__,
                                        #  self.in_channels, self.out_channels,
                                         self.K)

class WLGCN(torch.nn.Module):
    def __init__(self, input_size, output_size, K=8, dec_l=1, hidden_size=512, dropout=0.2, slope=0.2):
        super(WLGCN, self).__init__()
        self.conv1 = WLGCN_vanilla(K=K)
        self.fc1 = torch.nn.Linear(input_size * (K + 1), hidden_size)
        self.bn = torch.nn.BatchNorm1d(hidden_size)
        self.dropout1 = torch.nn.Dropout(p=dropout)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)
        self.negative_slope = slope

        if dec_l == 1:
            self.decoder = torch.nn.Linear(output_size, input_size)
        else:
            self.decoder = torch.nn.Sequential(
                torch.nn.Linear(output_size, output_size),
                torch.nn.ReLU(),
                torch.nn.Linear(output_size, input_size)
            )
        
    def forward(self, feature, edge_index, edge_weight=None):
        x = self.conv1(feature, edge_index, edge_weight)
        x = F.leaky_relu(self.fc1(x), negative_slope=self.negative_slope)
        x = self.bn(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        
        r = self.decoder(x)
        x = F.normalize(x, p=2, dim=1)
        return x, r

class GCN(Module):
    def __init__(self, in_feat, out_feat, dropout=0.0, act='relu'):
        super(GCN, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = get_activation(act)
        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight)
        x = torch.spmm(adj, x)
        x = self.act(x)

        return x

class AttentionLayer(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(AttentionLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat

        self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_omega)
        torch.nn.init.xavier_uniform_(self.u_omega)

    def forward(self, *inputs):
        
        emb = [torch.unsqueeze(input_, dim=1) for input_ in inputs]
        self.emb = torch.cat(emb, dim=1)

        self.v = torch.tanh(torch.matmul(self.emb, self.w_omega))
        self.vu = torch.matmul(self.v, self.u_omega)
        self.alpha = F.softmax(torch.squeeze(self.vu) + 1e-6)

        emb_combined = torch.matmul(torch.transpose(self.emb, 1, 2), torch.unsqueeze(self.alpha, -1))

        return torch.squeeze(emb_combined), self.alpha

class GCN_(nn.Module):
    def __init__(self, in_channels, out_channels, act='relu'):
        super(GCN_, self).__init__()
        self.conv1 = GCNConv(in_channels, out_channels)
        self.act = get_activation(act)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.act(x)
        return x

def ScaledDotProductAttention(Q, K):
    Q_norm = F.normalize(Q, dim=-1)
    K_norm = F.normalize(K, dim=-1)
    scores = torch.matmul(Q_norm, K_norm.transpose(-1, -2)) / np.sqrt(Q_norm.shape[1])
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, K)

# class EfficientAdditiveAttention(nn.Module):
#     def __init__(self, embed_dim):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.global_query = nn.Parameter(torch.randn(embed_dim))  
#         self.Proj = nn.Linear(embed_dim, embed_dim)

#     def forward(self, Q, V):
#         Q_norm = F.normalize(Q, dim=1)
#         V_norm = F.normalize(V, dim=1)
#         energy = torch.tanh(Q_norm + V_norm)
#         scores = torch.einsum("ij,j->i", energy, self.global_query) 
#         attn_weights = F.softmax(scores, dim=0)  
#         attn_weights = attn_weights.unsqueeze(-1) 
#         attn_output = attn_weights * V 
#         attn_output = self.Proj(attn_output)
#         return attn_output
    

class EfficientAdditiveAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.global_query = nn.Parameter(torch.randn(embed_dim))  
        self.Proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, Q, V):
        Q_norm = F.normalize(Q, dim=1)
        V_norm = F.normalize(V, dim=1)
        energy = torch.tanh(Q_norm + V_norm)
        scores = torch.einsum("ij,j->i", energy, self.global_query) / (self.embed_dim ** 0.5)  
        attn_weights = F.softmax(scores, dim=0)  
        attn_weights = attn_weights.unsqueeze(-1) 
        attn_output = attn_weights * V 
        attn_output = self.Proj(attn_output)
        return attn_output
    