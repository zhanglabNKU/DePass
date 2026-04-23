import datetime
import os
import random
import re
import time
import warnings
import numpy as np
import pandas as pd
import scipy
import torch
import scanpy as sc

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    
def select_feature_inter(adata_raw1,adata_raw2, hvg_num=3000):
        adata1 = adata_raw1.copy()
        adata2 = adata_raw2.copy()
        adata1.obs['batch'] = 'train'
        adata2.obs['batch'] = 'test'

        adata = sc.concat([adata1, adata2], axis=0, join='inner')
        adata.var_names_make_unique()
        adata.obs_names_make_unique()
    
        if hvg_num and len(adata.var_names) > hvg_num:
            sc.pp.highly_variable_genes(adata,flavor='seurat_v3',batch_key='batch', n_top_genes=hvg_num) 
            adata = adata[:, adata.var['highly_variable']]
              
        return adata1[:,adata.var_names], adata2[:,adata.var_names]


def preprocess_rna_inter(adata_raw, to_dense=True, norm=True):
    adata = adata_raw.copy()
    adata.var_names_make_unique()
    adata.obs_names_make_unique()

    if to_dense: 
        adata = todense(adata)
     
    if norm: 
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.scale(adata)
        
    return adata

      
def preprocess_pro_inter(adata_raw, to_dense=True, norm=True):
    adata = adata_raw.copy()
    adata.var_names_make_unique()
    adata.obs_names_make_unique()

    if to_dense: 
        adata = todense(adata)

    if norm:
       adata = clr_normalize_each_cell(adata)
       sc.pp.scale(adata) 
       
    return adata



def todense(adata_raw):
    adata = adata_raw.copy()
    if isinstance(adata.X, scipy.sparse.csr.csr_matrix) or isinstance(adata.X, scipy.sparse.csc.csc_matrix):
        dense_matrix = adata.X.todense()
        adata.X = np.array(dense_matrix)
    else:
        warnings.warn("adata.X is not of type csr_matrix")
 
    return adata



def clr_normalize_each_cell(adata, inplace=True):
    """
    Normalize each cell's protein counts using Centered Log-Ratio (CLR) normalization,
    following the approach used in Seurat and SpatialGLUE.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object where `.X` stores raw count data (e.g., protein abundance).
    
    inplace : bool, optional
        Whether to modify the input `adata` in place. If True, the normalization will overwrite `adata.X`. If False, a normalized copy of `adata` is returned. Default is True.

    Returns
    -------
    adata : AnnData
        The AnnData object with CLR-normalized `.X`. If `inplace=True`, returns the modified input object; if `inplace=False`, returns a new normalized copy.
    """
    
    def seurat_clr(x):
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    if not inplace:
        adata = adata.copy()
    
    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.X.A if scipy.sparse.issparse(adata.X) else np.array(adata.X))
    )
    return adata  




def saveResults(args, test_imputed_pro, test_true_pro, idx, start_time, obs_names=None, 
                result_path='result/rna_pro/intra/result_'):
    script_directory = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.abspath(os.path.join(script_directory, '../'))

    mode = args.mode

    result_path = f'result/{mode}/rna_pro/intra/result_'
    
    if hasattr(args, 'Pro_path_train') and args.Pro_path_train is not None:
        result_path = f'result/{mode}/rna_pro/inter/result_' 
        dataset_flag = getDatasetflagbyPath(args.Pro_path_train) + "_to_" + getDatasetflagbyPath(args.Pro_path_test)
    elif hasattr(args, 'Pro_path') and args.Pro_path is not None:
        dataset_flag = getDatasetflagbyPath(args.Pro_path)
    else:
        dataset_flag = "unknown"
    
    folder_path = os.path.join(base_path, result_path + args.method_flag, dataset_flag)
    os.makedirs(folder_path, exist_ok=True)

    test_imputed_pro = pd.DataFrame(test_imputed_pro, columns=idx, index=obs_names)
    test_true_pro = pd.DataFrame(test_true_pro, columns=idx, index=obs_names)
    
    pred_csv_path = os.path.join(folder_path, f'{args.repeat}_y_pred.csv')
    true_csv_path = os.path.join(folder_path, f'{args.repeat}_y_truth.csv')
    
    test_imputed_pro.to_csv(pred_csv_path)
    print(f"[INFO] → Saving predicted protein to: {pred_csv_path}")
    
    test_true_pro.to_csv(true_csv_path)
    print(f"[INFO] → Saving true protein to: {true_csv_path}")

    args_dict = vars(args)
    args_file_path = os.path.join(folder_path, f'args{args.repeat}.txt')
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cost_time = time.time() - start_time

    with open(args_file_path, 'w') as file:
        for k, v in args_dict.items():  
            file.write(f'{k}:{v}\n')
        file.writelines([
            '-' * 80 + '\n',
            f'Performance in repeat_{args.repeat} costTime: {cost_time:.4f}s\n',
            f'Current Time: {current_time}\n'
        ])
    print(f"[INFO] → Saving experiment args to: {args_file_path}")


def saveH5adResults(args, pred_adata, true_adata, result_path='result/rna_pro/intra/result_'):
    script_directory = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.abspath(os.path.join(script_directory, '../'))

    mode = args.mode
    result_path = f'result/{mode}/rna_pro/intra/result_'
    
    if hasattr(args, 'Pro_path_train') and args.Pro_path_train is not None:
        result_path = f'result/{mode}/rna_pro/inter/result_' 
        dataset_flag = getDatasetflagbyPath(args.Pro_path_train) + "_to_" + getDatasetflagbyPath(args.Pro_path_test)
    elif hasattr(args, 'Pro_path') and args.Pro_path is not None:
        dataset_flag = getDatasetflagbyPath(args.Pro_path)
    else:
        dataset_flag = "unknown"

    folder_path = os.path.join(base_path, result_path + args.method_flag, dataset_flag)
    os.makedirs(folder_path, exist_ok=True)
    
    pred_h5ad_path = os.path.join(folder_path, f'{args.repeat}_y_pred.h5ad')
    true_h5ad_path = os.path.join(folder_path, f'{args.repeat}_y_truth.h5ad')
    
    pred_adata.write_h5ad(pred_h5ad_path)
    print(f"[INFO] → Saving predicted protein to: {pred_h5ad_path}")
    
    true_adata.write_h5ad(true_h5ad_path)
    print(f"[INFO] → Saving true protein to: {true_h5ad_path}")





def getDatasetflagbyPath(path):
    parts = re.split(r'[\\/]+', path.strip('/\\'))

    parts = [p for p in parts if p]

    dataset_flag = parts[-2] if len(parts) >= 2 else None
    return dataset_flag

