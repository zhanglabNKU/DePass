import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
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
    