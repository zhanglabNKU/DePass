
# The source code is borrowed from [GitHub: https://github.com/jlakkis/sciPENN] 

import argparse
import datetime
import os
import random
import sys
import time
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from anndata import AnnData
from torch import zeros_like

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.append(project_root)
from methods.utils import *
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for each GPU training (default: 16)')
parser.add_argument('--test_batch_size', type=int, default=128,
                    help='input batch size for testing (default: 32)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 500)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1105,
                    help='random seed (default: 1105)')
parser.add_argument('--repeat', type=int, default=1,
                    help='random seed (default: 1105)')
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--frac_finetune_test', type=float, default=0.1,
                    help='test set ratio')
parser.add_argument('--enc_max_seq_len', type=int, default=20000,
                    help='sequence length of encoder')
parser.add_argument('--dec_max_seq_len', type=int, default=224,
                    help='sequence length of decoder')
parser.add_argument('--fix_set', action='store_false',
                    help='fix or disordering set')
parser.add_argument('--resume', default=False, help='resume training from checkpoint')
parser.add_argument('--RNA_path_train', default='/dataset/rna_pro/inter/dataset12_dataset23/dataset12/adata_RNA.h5ad',
                    help='path for loading the rna')
parser.add_argument('--Pro_path_train', default='/dataset/rna_pro/inter/dataset12_dataset23/dataset12/adata_ADT.h5ad',
                    help='path for loading the protein')    
parser.add_argument('--RNA_path_test', default='/dataset/rna_pro/inter/dataset12_dataset23/dataset23/adata_RNA.h5ad',
                    help='path for loading the rna_test')
parser.add_argument('--Pro_path_test', default='/dataset/rna_pro/inter/dataset12_dataset23/dataset23/adata_ADT.h5ad',
                    help='path for loading the protein_test')    
parser.add_argument('--method_flag', default='sciPENN',
                help='method_flag')   
parser.add_argument('--mode', default='sc')
   
args = parser.parse_args()

use_cuda = not args.no_cuda and torch.cuda.is_available()
setup_seed(args.seed+args.repeat)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    loss2 = nn.CosineSimilarity(dim=0, eps=1e-8)
    train_loss = 0
    train_ccc = 0
    y_hat_all, y_all = [], []
    for batch_idx, (x, y) in enumerate(train_loader):
        pro_mask = torch.tensor(y[:,2].tolist()).bool().cuda()
        x = torch.tensor(x[:,0].tolist(), dtype=torch.float32).cuda()
        y = torch.tensor(y[:,0].tolist(), dtype=torch.float32).cuda()

        #--- Prediction ---#
        optimizer.zero_grad()
        y_hat = model(x)

        #--- Compute Performance Metric ---#
        y_hat = torch.squeeze(y_hat)
        y_hat = torch.where(torch.isnan(y), torch.full_like(y_hat, 0), y_hat)
        y = torch.where(torch.isnan(y), torch.full_like(y, 0), y)
        loss = F.mse_loss(y_hat[pro_mask], y[pro_mask])
        train_loss += loss.item()
        
        train_ccc += loss2(y_hat[pro_mask], y[pro_mask]).item()
        if device != 'cpu':
            y_hat = y_hat.detach().cpu()
            y = y.detach().cpu()
        if len(x) >1:
            y_hat_all.extend(y_hat.numpy().tolist())
            y_all.extend(y.numpy().tolist())
        else:
            y_hat_all.append(y_hat.numpy().tolist())
            y_all.append(y.numpy().tolist())
        loss.backward()
        optimizer.step()

    train_loss /= len(train_loader)
    train_ccc /= len(train_loader)
    
    return train_loss, train_ccc
    

def test(model, device, test_loader):
    model.eval()
    loss2 = nn.CosineSimilarity(dim=0, eps=1e-8)
    test_loss = 0
    test_ccc = 0
    y_hat_all, y_all = [], []
    with torch.no_grad():
        for x, y in test_loader:
            #--- Extract Feature ---#
            pro_mask = torch.tensor(y[:,2].tolist()).bool().cuda()
            x = torch.tensor(x[:,0].tolist(), dtype=torch.float32).cuda()
            y = torch.tensor(y[:,0].tolist(), dtype=torch.float32).cuda()

            #--- Prediction ---#
            y_hat = model(x)
            #--- Compute Performance Metric ---#
            # remove effect of Nan
            count = (torch.isnan(y)==True).sum()
            y_hat = torch.squeeze(y_hat)
            y_hat = torch.where(torch.isnan(y), torch.full_like(y_hat, 0), y_hat)
            y = torch.where(torch.isnan(y), torch.full_like(y, 0), y)
            if device != 'cpu':
                y_hat = y_hat.detach().cpu()
                y = y.detach().cpu()
                pro_mask = pro_mask.detach().cpu()
            if len(x) >1:
                y_hat_all.extend(y_hat.numpy().tolist())
                y_all.extend(y.numpy().tolist())
            else:
                y_hat_all.append(y_hat.numpy().tolist())
                y_all.append(y.numpy().tolist())



    return y_hat_all, y_all




from Layers import FF_Block, Input_Block
from torch.nn import Linear, Module, RNNCell


class sciPENN_Model(Module):
    def __init__(self, p_mod1, p_mod2):
        super(sciPENN_Model, self).__init__()
        
        h_size, drop_rate = 512, 0.25
        
        self.RNNCell = RNNCell(h_size, h_size)   
        self.input_block = Input_Block(p_mod1, h_size, drop_rate, drop_rate) 
        self.skip_1 = FF_Block(h_size, drop_rate)
        self.skip_2 = FF_Block(h_size, drop_rate)
        self.skip_3 = FF_Block(h_size, drop_rate)
        self.mod2_out = Linear(h_size, p_mod2)
              
    def forward(self, x):
        x = self.input_block(x)
        h = self.RNNCell(x, zeros_like(x))
        
        x = self.skip_1(x)
        h = self.RNNCell(x, h)
        
        x = self.skip_2(x)
        h = self.RNNCell(x, h)
        
        x = self.skip_3(x)
        h = self.RNNCell(x, h)
        
        return self.mod2_out(h)


def main():
    scRNA_adata_train = sc.read_h5ad(args.RNA_path_train)
    scP_adata_train =   sc.read_h5ad(args.Pro_path_train)
    scRNA_adata_test =  sc.read_h5ad(args.RNA_path_test)
    scP_adata_test =    sc.read_h5ad(args.Pro_path_test)

    scRNA_adata_train,scRNA_adata_test = select_feature_inter(scRNA_adata_train,scRNA_adata_test,hvg_num=3000)
    scP_adata_train,scP_adata_test = select_feature_inter(scP_adata_train,scP_adata_test,hvg_num=False)
 
    train_rna = scRNA_adata_train
    test_rna = scRNA_adata_test
    train_protein = scP_adata_train
    test_protein = scP_adata_test


    train_rna = preprocess_rna_inter(train_rna, norm=False)
    train_protein = preprocess_pro_inter(train_protein, norm=False)
    test_rna = preprocess_rna_inter(test_rna, norm=False)
    test_protein = preprocess_pro_inter(test_protein, norm=False)


    args.enc_max_seq_len = train_rna.n_vars
    args.dec_max_seq_len = train_protein.n_vars
    model = sciPENN_Model(args.enc_max_seq_len, args.dec_max_seq_len)
    
    start_epoch = 0
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 4,
                       'shuffle': False,
                       'prefetch_factor': 2,
                       'pin_memory': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)



    my_trainset = SCDataset(train_rna, train_protein)
    my_testset = SCDataset(test_rna, test_protein)
    train_loader = torch.utils.data.DataLoader(my_trainset, **train_kwargs, drop_last=False)
    test_loader = torch.utils.data.DataLoader(my_testset, **test_kwargs, drop_last=False)

    start_time = time.time()
    best_test_loss, counter = float('inf'), 0
    for epoch in range(start_epoch+1, args.epochs + 1):
        train_loss, train_ccc = train(args, model, device, train_loader, optimizer, epoch)
        print('In epoch %d: train_loss%.4f, train_ccc%.4f' % (epoch, train_loss, train_ccc))

    test_imputed_pro, test_true_pro = test(model, device, test_loader)

    saveResults(args, test_imputed_pro, test_true_pro, test_protein.var.index.tolist(), start_time, obs_names=test_protein.obs_names.to_list())


    true_pro_adata = test_protein.copy()
    true_pro_adata.X = test_true_pro
    pred_pro_adata = AnnData(
        X=np.array(test_imputed_pro), 
        obs=true_pro_adata.obs.copy(),
        var=pd.DataFrame(index=true_pro_adata.var.index)
    )
    saveH5adResults(args, pred_pro_adata, true_pro_adata)

if __name__ == '__main__':

    main()