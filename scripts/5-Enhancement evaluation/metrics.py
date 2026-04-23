import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.sparse as sp
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Union
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from anndata import AnnData




def calculate_coefficient_of_variation(expr):
    expr = expr[~np.isnan(expr)]
    if len(expr) < 2:
        return np.nan
    mean_val = np.mean(expr)
    if mean_val == 0:
        return np.nan
    return stats.variation(expr, ddof=1)



def calculate_variance_ratio(domain_means, domain_vars):
    if len(domain_means) < 2:
        return np.nan
    between_var = np.var(domain_means, ddof=1)
    within_var = np.mean(domain_vars) if domain_vars else np.nan
    if np.isnan(within_var) or within_var == 0:
        return np.nan
    return between_var / within_var



def compute_domain_metrics(adata, domain_col="cell_type", norm="min-max"):
    results = []
    domains = adata.obs[domain_col].unique()

    adata_norm = adata.copy()
    
    if sp.issparse(adata_norm.X):
        expr_matrix = adata_norm.X.toarray()
    else:
        expr_matrix = adata_norm.X.copy()
    
    if norm == "min-max":
        scaler = MinMaxScaler()
        expr_matrix_normed = scaler.fit_transform(expr_matrix)
    elif norm == "z-score":
        scaler = StandardScaler()
        expr_matrix_normed = scaler.fit_transform(expr_matrix)
    elif norm is None:
        expr_matrix_normed = expr_matrix
    else:
        raise ValueError(f"Unsupported normalization type: {norm}, choose 'min-max' or 'z-score'")

    for i, gene in enumerate(adata.var_names):
        gene_domain_stats = []
        domain_means = []
        domain_vars = []

        gene_global_expr = expr_matrix_normed[:, i].flatten()
        
        for domain in domains:
            idx = adata.obs[domain_col] == domain
            if idx.sum() < 2:
                continue

            expr_normed = gene_global_expr[idx]
            expr_normed = expr_normed[~np.isnan(expr_normed)]
            
            if len(expr_normed) < 2:
                continue

            coeff_var = calculate_coefficient_of_variation(expr_normed)
            domain_var = np.var(expr_normed, ddof=1)
            domain_mean = np.mean(expr_normed)

            gene_domain_stats.append({
                "gene": gene,
                "domain": domain,
                "coefficient_of_variation": coeff_var
            })
            domain_means.append(domain_mean)
            domain_vars.append(domain_var)

        variance_ratio = calculate_variance_ratio(domain_means, domain_vars)

        for record in gene_domain_stats:
            record["variance_ratio"] = variance_ratio

        results.extend(gene_domain_stats)

    return pd.DataFrame(results)



def calculate_logfc(
    adata: AnnData,
    target_gene: str,
    target_group: str,
    groupby: str = "DePass",
    method: str = "wilcoxon",
    n_genes: int = 10,
    save_path: str = "results"
):
    os.makedirs(save_path, exist_ok=True)
    
    adata.obs[groupby] = adata.obs[groupby].astype('category')
    adata.X = MinMaxScaler().fit_transform(adata.X)
    sc.tl.rank_genes_groups(adata, groupby=groupby, method=method, use_raw=False)
    
    sc.pl.rank_genes_groups_dotplot(adata, groupby=groupby, n_genes=n_genes, show=False, dendrogram=False)
    plt.savefig(f"{save_path}/rank_genes.png", dpi=300, bbox_inches="tight")
    plt.close()

    names = adata.uns['rank_genes_groups']['names']
    lfc = adata.uns['rank_genes_groups']['logfoldchanges']
    idx = np.flatnonzero(names[target_group] == target_gene)[0]
    logfc_val = float(lfc[target_group][idx])

    df = pd.DataFrame({
        "Gene": [target_gene],
        "Group": [target_group],
        "logFC": [logfc_val]
    })
    df.to_csv(f"{save_path}/logfc.csv", index=False)
    
    return logfc_val, df