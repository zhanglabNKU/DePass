
# The source code is borrowed from [GitHub: https://github.com/BioX-NKU/scButterfly] 

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from scipy import sparse
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.append(project_root)
from anndata import AnnData
from scButterfly.train_model_cite import Model
from methods.utils import *
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='input batch size for each GPU training (default: 16)')
parser.add_argument('-test_batch_size-', type=int, default=32,
                    help='input batch size for testing (default: 32)')
parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                    help='learning rate (default: 1.0)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1105,
                    help='random seed (default: 1105)')
parser.add_argument('--repeat', type=int, default=1,
                    help='random seed (default: 1105)')
parser.add_argument('--frac_finetune_test', type=float, default=0.1,
                    help='test set ratio') 
parser.add_argument('--RNA_path_train', default='/dataset/rna_pro/inter/dataset12_dataset23/dataset12/adata_RNA.h5ad',
                    help='path for loading the rna')
parser.add_argument('--Pro_path_train', default='/dataset/rna_pro/inter/dataset12_dataset23/dataset12/adata_ADT.h5ad',
                    help='path for loading the protein')    
parser.add_argument('--RNA_path_test', default='/dataset/rna_pro/inter/dataset12_dataset23/dataset23/adata_RNA.h5ad',
                    help='path for loading the rna_test')
parser.add_argument('--Pro_path_test', default='/dataset/rna_pro/inter/dataset12_dataset23/dataset23/adata_ADT.h5ad',
                    help='path for loading the protein_test')      
parser.add_argument('--method_flag', default='scButterfly',
                    help='method_flag')    
 
parser.add_argument('--mode', default='sc')
   
args = parser.parse_args()

use_cuda = not args.no_cuda and torch.cuda.is_available()
setup_seed(args.seed + args.repeat)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    scRNA_adata = train_rna.concatenate(test_rna, batch_key=None)
    scP_adata = train_protein.concatenate(test_protein, batch_key=None)
    

    train_index, val_index = train_test_split(
        range(len(train_rna)), 
        test_size=0.1111,  # 0.1 / 0.9 ≈ 0.1111
        random_state=42
    )

    test_index = np.arange(len(train_rna), len(train_rna) + len(test_rna))

    start_time = time.time()

    RNA_input_dim = scRNA_adata.X.shape[1]
    ADT_input_dim = scP_adata.X.shape[1]
    R_kl_div = 1 / RNA_input_dim * 20
    A_kl_div = 1 / 150
    kl_div = R_kl_div + A_kl_div


    scP_adata.X = sparse.csr_matrix(scP_adata.X)
    scRNA_adata.X = sparse.csr_matrix(scRNA_adata.X)

    model = Model(
        R_encoder_nlayer = 2, 
        A_encoder_nlayer = 2,
        R_decoder_nlayer = 2, 
        A_decoder_nlayer = 2,
        R_encoder_dim_list = [RNA_input_dim, 256, 128],
        A_encoder_dim_list = [ADT_input_dim, 128, 128],
        R_decoder_dim_list = [128, 256, RNA_input_dim],
        A_decoder_dim_list = [128, 128, ADT_input_dim],
        R_encoder_act_list = [nn.LeakyReLU(), nn.LeakyReLU()],
        A_encoder_act_list = [nn.LeakyReLU(), nn.LeakyReLU()],
        R_decoder_act_list = [nn.LeakyReLU(), nn.LeakyReLU()],
        A_decoder_act_list = [nn.LeakyReLU(), nn.Identity()],
        translator_embed_dim = 128, 
        translator_input_dim_r = 128,
        translator_input_dim_a = 128,
        translator_embed_act_list = [nn.LeakyReLU(), nn.LeakyReLU(), nn.LeakyReLU()],
        discriminator_nlayer = 1,
        discriminator_dim_list_R = [128],
        discriminator_dim_list_A = [128],
        discriminator_act_list = [nn.Sigmoid()],
        dropout_rate = 0.1,
        R_noise_rate = 0.5,
        A_noise_rate = 0,
        chrom_list = [],
        logging_path = None,
        RNA_data = scRNA_adata,
        ATAC_data = scP_adata
    )

    train_id_r = train_index.copy()
    train_id_a = train_index.copy()
    validation_id_r = val_index.copy()
    validation_id_a = val_index.copy()
    test_id_r = test_index.copy()
    test_id_a = test_index.copy()

    model.train(
        R_encoder_lr = 0.001,
        A_encoder_lr = 0.001,
        R_decoder_lr = 0.001,
        A_decoder_lr = 0.001,
        R_translator_lr = 0.001,
        A_translator_lr = 0.001,
        translator_lr = 0.001,
        discriminator_lr = 0.005,
        R2R_pretrain_epoch = 100,
        A2A_pretrain_epoch = 100,
        lock_encoder_and_decoder = False,
        translator_epoch = 200,
        patience = 50,
        batch_size = 64,
        r_loss = nn.MSELoss(size_average=True),
        a_loss = nn.MSELoss(size_average=True),
        d_loss = nn.BCELoss(size_average=True),
        loss_weight = [1, 2, 1, R_kl_div, A_kl_div, kl_div],
        train_id_r = train_id_r,
        train_id_a = train_id_a,
        validation_id_r = validation_id_r, 
        validation_id_a = validation_id_a, 
        output_path = None,
        seed = args.seed + args.repeat, #19193,
        kl_mean = True,
        R_pretrain_kl_warmup = 50,
        A_pretrain_kl_warmup = 50,
        translation_kl_warmup = 50,
        load_model = None,
        logging_path = None
    )

    A2R_predict, R2A_predict = model.test(
        test_id_r = test_id_r,
        test_id_a = test_id_a, 
        model_path = None,
        load_model = False,
        output_path = None,
        test_cluster = False,
        test_figure = False,
        output_data = False,
        return_predict = True
    )
    
    test_imputed_pro= R2A_predict.X.toarray()
    true_pro_adata = test_protein.copy()
    test_true_pro=true_pro_adata.X
    true_pro_adata.X = test_true_pro
    
    pred_pro_adata = AnnData(X = test_imputed_pro, 
                            obs = true_pro_adata.obs.copy(),
                            var = pd.DataFrame(index=true_pro_adata.var.index))
    saveH5adResults(args, pred_pro_adata, true_pro_adata)
    saveResults(args, test_imputed_pro, test_true_pro, true_pro_adata.var.index.tolist(), start_time, obs_names = true_pro_adata.obs.index.tolist())

if __name__ == '__main__':
    main()
    