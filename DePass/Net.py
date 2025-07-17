import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from typing import Any, Optional, Tuple
from torch.autograd import Function
from .layer import *

class DePassAE(Module):
    def __init__(self, dim_input1, dim_input2, mlplayer1, mlplayer2, dim, dropout=0.2, act='elu'):
        super(DePassAE, self).__init__() 

        self.mlplayer1=mlplayer1
        self.mlplayer2=mlplayer2
        self.dim=dim
        self.act = act
        self.dropout = dropout

        self.mlp_encoder1 = mlp_encoder(dim_input1, self.mlplayer1,act=self.act)
        self.mlp_encoder2 = mlp_encoder(dim_input2, self.mlplayer2,act=self.act)

        self.gcn_encoder_s1 = GCN(self.mlplayer1[-1],self.dim)
        self.gcn_encoder_s2 = GCN(self.mlplayer2[-1],self.dim) 
        self.gcn_encoder_con = GCN(self.mlplayer1[-1]+self.mlplayer1[-1],self.dim) 

        # self._gcn_encoder_s1 =  GCN(self.dim,self.dim)
        # self._gcn_encoder_s2 =  GCN(self.dim,self.dim) 
        # self._gcn_encoder_con = GCN(self.dim,self.dim) 

        # self.__gcn_encoder_s1 =  GCN(self.dim,self.dim)
        # self.__gcn_encoder_s2 =  GCN(self.dim,self.dim) 
        # self.__gcn_encoder_con = GCN(self.dim,self.dim) 

        self.Eff_Att1 = EfficientAdditiveAttention(self.dim)
        self.Eff_Att2 = EfficientAdditiveAttention(self.dim)

        self.atten_cross = AttentionLayer(2 * self.dim, 2 * self.dim)
        # self.lin1 = nn.Linear(2 * self.dim,32)
        # self.elu = nn.ELU()
        # self.lin2 = nn.Linear(64,32)
        # self.atten_cross = AttentionLayer(1 * self.dim, 1 * self.dim)

        # self._gcn_decoder1 = GCN(2 * self.dim, 2 * self.dim)
        # self._gcn_decoder2 = GCN(2 * self.dim, 2 * self.dim)
        # self.__gcn_decoder1 = GCN(2 * self.dim, 2 * self.dim)
        # self.__gcn_decoder2 = GCN(2 * self.dim, 2 * self.dim)

        self.gcn_decoder1 = GCN(2 * self.dim, dim_input1)
        self.gcn_decoder2 = GCN(2 * self.dim, dim_input2)

        # self.gcn_decoder1 = GCN(32, dim_input1)
        # self.gcn_decoder2 = GCN(32, dim_input2)

    def mlp_forward(self,x1,z1,x2,z2):
        x1 = self.mlp_encoder1(x1)
        z1 = self.mlp_encoder1(z1)
        x2 = self.mlp_encoder2(x2)
        z2 = self.mlp_encoder2(z2)

        e1 = torch.mean(torch.stack([x1, z1], dim=0), dim=0)
        e2 = torch.mean(torch.stack([x2, z2], dim=0), dim=0)

        return e1, e2
    
    def gcn_forward(self, e1_batch, e2_batch, adj_shared_batch, adj1_batch, adj2_batch):
       
        E_fusion = torch.cat([e1_batch, e2_batch], dim=-1)
        s1 = self.gcn_encoder_s1(e1_batch, adj1_batch)
        s2 = self.gcn_encoder_s2(e2_batch, adj2_batch)
        f =  self.gcn_encoder_con(E_fusion, adj_shared_batch)

        # s1 = self._gcn_encoder_s1(s1, adj1_batch)
        # s2 = self._gcn_encoder_s2(s2, adj2_batch)
        # f =  self._gcn_encoder_con(f, adj_shared_batch)

        # s1 = self.__gcn_encoder_s1(s1, adj1_batch)
        # s2 = self.__gcn_encoder_s2(s2, adj2_batch)
        # f =  self.__gcn_encoder_con(f, adj_shared_batch)

        s1 = self.Eff_Att1(f,s1) 
        s2 = self.Eff_Att2(f,s2) 
     

        z1 = torch.cat([s1, f], dim=-1)
        z2 = torch.cat([s2, f], dim=-1)

        # z1=s1
        # z2=s2
        z, alpha_omics_1_2 = self.atten_cross(z1, z2)
        # z = self.lin1(z)
        # z = self.elu(z)
        # z = self.lin2(z)

        # emb_recon1 = self._gcn_decoder1(z, adj_shared_batch)
        # emb_recon2 = self._gcn_decoder2(z, adj_shared_batch)

        # emb_recon1 = self.__gcn_decoder1(z, adj_shared_batch)
        # emb_recon2 = self.__gcn_decoder2(z, adj_shared_batch)

        emb_recon1 = self.gcn_decoder1(z, adj_shared_batch)
        emb_recon2 = self.gcn_decoder2(z, adj_shared_batch)

        # emb_reconz1 = self.gcn_decoder1(z1, adj_shared_batch)
        # emb_reconz2 = self.gcn_decoder2(z2, adj_shared_batch)

        results = {
            'z1': z1,
            'z2': z2,
            'z': z,
            'emb_recon1': emb_recon1,
            'emb_recon2': emb_recon2,
            'alpha': alpha_omics_1_2,
        }

        return results

    def gcn_inference(self, e1, e2, adj_shared, adj1, adj2):
        E_fusion = torch.cat([e1, e2], dim=-1)
        
        s1 = self.gcn_encoder_s1(e1, adj1)
        s2 = self.gcn_encoder_s2(e2, adj2)
        f = self.gcn_encoder_con(E_fusion, adj_shared)

        s1 = self.Eff_Att1(f,s1) 
        s2 = self.Eff_Att2(f,s2) 
        # s1 = ScaledDotProductAttention(f,s1) 
        # s2 = ScaledDotProductAttention(f,s2) 

        z1 = torch.cat([s1, f], dim=-1)
        z2 = torch.cat([s2, f], dim=-1)
        # z1=s1
        # z2=s2

        z, alpha_omics_1_2 = self.atten_cross(z1, z2)

        results = {
            'z1': z1,
            'z2': z2,
            'z': z,
            'alpha': alpha_omics_1_2,
        }

        return results

class DePassAE1(Module):
    def __init__(self, dim_input1, mlplayer1, dim, dropout=0.2, act='elu'):
        super(DePassAE1, self).__init__() 
        self.mlplayer1=mlplayer1
        self.dim=dim
        self.act = act
        self.dropout = dropout
        self.mlp_encoder1 = mlp_encoder(dim_input1, self.mlplayer1,act=self.act)
        self.gcn_encoder_s1 = GCN(self.mlplayer1[-1],self.dim)
        self.gcn_encoder_con = GCN(self.mlplayer1[-1],self.dim) 
        self.Eff_Att1 = EfficientAdditiveAttention(self.dim)
        self.gcn_decoder1 = GCN(2 * self.dim, dim_input1)

    def mlp_forward(self,x1,z1):
        x1 = self.mlp_encoder1(x1)
        z1 = self.mlp_encoder1(z1)
        e1 = torch.mean(torch.stack([x1, z1], dim=0), dim=0)
        return e1
    
    def gcn_forward(self, e1_batch, adj_shared_batch, adj1_batch):
        s1 = self.gcn_encoder_s1(e1_batch, adj1_batch)
        f =  self.gcn_encoder_con(e1_batch, adj_shared_batch)
        s1 = self.Eff_Att1(f,s1)
        z = torch.cat([s1, f], dim=-1) 
        emb_recon1 = self.gcn_decoder1(z, adj_shared_batch)
        results = {
            'z': z,
            'emb_recon1': emb_recon1
        }
        return results

    def gcn_inference(self, e1, adj_shared, adj1):
        s1 = self.gcn_encoder_s1(e1, adj1)
        f = self.gcn_encoder_con(e1, adj_shared)
        s1 = self.Eff_Att1(f,s1) 
        z = torch.cat([s1, f], dim=-1)

        results = {
            'z': z
        }

        return results

class DePassAE_va(Module):
    def __init__(self, dim_input_list, mlplayer_list, dim, dropout=0.2, act='elu'):
        super(DePassAE_va, self).__init__()
        self.num_modalities = len(dim_input_list)
        self.mlplayer_list = mlplayer_list
        self.dim = dim
        self.act = act
        self.dropout = dropout

        self.mlp_encoders = nn.ModuleList([
            mlp_encoder(dim_input_list[i], mlplayer_list[i], act=self.act)
            for i in range(self.num_modalities)
        ])

        self.gcn_encoders_s = nn.ModuleList([
            GCN(mlplayer_list[i][-1], self.dim)
            for i in range(self.num_modalities)
        ])

        fused_dim = sum(mlplayer_list[i][-1] for i in range(self.num_modalities))
        self.gcn_encoder_con = GCN(fused_dim, self.dim)

        self.eff_atts = nn.ModuleList([
            EfficientAdditiveAttention(self.dim) for _ in range(self.num_modalities)
        ])

        self.atten_cross = AttentionLayer(2 * self.dim, 2 * self.dim)

        self.gcn_decoders = nn.ModuleList([
            GCN(2 * self.dim, dim_input_list[i]) for i in range(self.num_modalities)
        ])

    def mlp_forward(self,xs,zs):
        es = []  
        for i in range(self.num_modalities):
            x = self.mlp_encoders[i](xs[i])
            z = self.mlp_encoders[i](zs[i])
            e = torch.mean(torch.stack([x, z], dim=0), dim=0)
            es.append(e)
        return es

    def gcn_forward(self, es, adj_shared, adjs):
   
        E_fusion = torch.cat(es, dim=-1)

        ss = []
        for i in range(self.num_modalities):
            s = self.gcn_encoders_s[i](es[i], adjs[i])
            ss.append(s)

        f = self.gcn_encoder_con(E_fusion, adj_shared)

        for i in range(self.num_modalities):
            ss[i] = self.eff_atts[i](f, ss[i])

        fused_features = []
        for i in range(self.num_modalities):
            fused_feature = torch.cat([ss[i], f], dim=-1)
            fused_features.append(fused_feature)

        z, alpha = self.atten_cross(*fused_features)


        emb_recons = []
        for i in range(self.num_modalities):
            emb_recon = self.gcn_decoders[i](z, adj_shared)
            emb_recons.append(emb_recon)

        results = {
            'z': z,
            'alpha': alpha,
            'emb_recons': emb_recons,
        }

        return results

    def gcn_inference(self, es, adj_shared, adjs):

        E_fusion = torch.cat(es, dim=-1)

        ss = []
        for i in range(self.num_modalities):
            s = self.gcn_encoders_s[i](es[i], adjs[i])
            ss.append(s)

        f = self.gcn_encoder_con(E_fusion, adj_shared)

        for i in range(self.num_modalities):
            ss[i] = self.eff_atts[i](f, ss[i])

        fused_features = []
        for i in range(self.num_modalities):
            fused_feature = torch.cat([ss[i], f], dim=-1)
            fused_features.append(fused_feature)

        z, alpha = self.atten_cross(*fused_features)

        results = {
            'z': z,
            'alpha': alpha,
        }

        for i in range(self.num_modalities):
            results[f'z{i+1}'] = ss[i]

        return results