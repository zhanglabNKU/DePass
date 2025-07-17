import numpy as np
import scipy
import os
import random
import torch
from torch.backends import cudnn
import anndata
import scanpy as sc
import torch_geometric

############# Graph Construction #############

def construct_knn_graph_hnsw(data: np.ndarray, k: int = 20, space: str = r'l2') -> torch.Tensor:
    r"""
    Construct a k-NN graph using the HNSW algorithm.

    Parameters:
        data (numpy.ndarray): Input data matrix (num_samples x dim).
        k (int): Number of nearest neighbors (default: 20).
        space (str): Distance metric (r'l2' for Euclidean distance, default).

    Returns:
        adj (torch.Tensor): Adjacency matrix of the k-NN graph (dense tensor).
    """

    import hnswlib
    from torch_geometric.utils import to_dense_adj
    data = data.astype(np.float32)
    num_samples, dim = data.shape
    p = hnswlib.Index(space=space, dim=dim)
    p.init_index(max_elements=num_samples, ef_construction=200, M=16)
    p.add_items(data)
    p.set_ef(50)
    indices, _ = p.knn_query(data, k=k)
    row_indices = np.repeat(np.arange(num_samples), k)
    col_indices = indices.flatten()
    edge_index = torch.tensor([row_indices, col_indices], dtype=torch.long)
    adj = to_dense_adj(edge_index)[0]
    return adj


import numpy as np
import torch
import hnswlib
from typing import Union

def construct_knn_graph_hnsw_sparse(
    data: np.ndarray,
    k: int = 20,
    space: str = 'l2',
    symmetric: bool = False,
    sparse_format: str = 'torch_coo'
) -> Union[torch.Tensor, scipy.sparse.spmatrix]:
    r"""
    Construct a k-NN graph using HNSW and return a sparse adjacency matrix.

    Parameters:
        data (np.ndarray): Input data matrix (num_samples x dim).
        k (int): Number of nearest neighbors.
        space (str): Distance metric ('l2' or 'ip').
        symmetric (bool): Whether to make the adjacency matrix symmetric (undirected graph).
        sparse_format (str): Output format ('coo', 'csr', 'torch_coo').

    Returns:
        adj (Union[torch.Tensor, scipy.sparse.spmatrix]): Sparse adjacency matrix.
    """
    # --- Step 1: Generate HNSW k-NN indices ---
    data = data.astype(np.float32)
    num_samples, dim = data.shape

    # Build HNSW index
    p = hnswlib.Index(space=space, dim=dim)
    p.init_index(max_elements=num_samples, ef_construction=200, M=16)
    p.add_items(data)
    p.set_ef(50)
    
    # Query k-NN
    indices, _ = p.knn_query(data, k=k)
    
    # --- Step 2: Generate edge indices ---
    row_indices = np.repeat(np.arange(num_samples), k)
    col_indices = indices.flatten()
    
    # Remove self-loops
    mask = row_indices != col_indices
    row_indices = row_indices[mask]
    col_indices = col_indices[mask]

    # --- Step 3: Symmetrize edges (if needed) ---
    if symmetric:
        # Add reverse edges
        row_indices_sym = np.concatenate([row_indices, col_indices])
        col_indices_sym = np.concatenate([col_indices, row_indices])
        
        # Remove duplicates
        edges = np.vstack([row_indices_sym, col_indices_sym]).T
        edges = np.unique(edges, axis=0)
        row_indices, col_indices = edges[:, 0], edges[:, 1]

    # --- Step 4: Build sparse adjacency matrix ---
    if sparse_format in ['coo', 'csr']:
        # SciPy sparse matrix
        import scipy.sparse as sp
        adj = sp.coo_matrix(
            np.ones_like(row_indices), 
            (row_indices, col_indices),
            shape=(num_samples, num_samples)
        )
        if sparse_format == 'csr':
            adj = adj.tocsr()
    
    elif sparse_format == 'torch_coo':
        # PyTorch sparse tensor (COO format)
        indices = torch.vstack([
            torch.LongTensor(row_indices),
            torch.LongTensor(col_indices)
        ])
        adj = torch.sparse_coo_tensor(
            indices, 
            torch.ones(indices.shape[1]), 
            size=(num_samples, num_samples)
        )
    
    else:
        raise ValueError(f"Invalid sparse_format: {sparse_format}")

    return adj

def to_undirected_geo_data(adj_shared, node_index=None) -> torch_geometric.data.Data:
    r"""
    Converts the input adjacency matrix (sparse matrix, sparse tensor, dense tensor, or dense NumPy array)
    into an undirected graph's PyTorch Geometric Data object.

    Parameters:
        adj_shared: The input adjacency matrix (sparse matrix, sparse tensor, dense tensor, or dense NumPy array).
        node_index (optional): Node indices.

    Returns:
        undirected_geo (torch_geometric.data.Data): The converted undirected graph Data object.
    """
    from torch_geometric.data import Data
    from torch_geometric.transforms import ToUndirected
    from torch_geometric.utils import dense_to_sparse, from_scipy_sparse_matrix
    import scipy.sparse as sp

    if sp.issparse(adj_shared):
        edge_index, edge_attr = from_scipy_sparse_matrix(adj_shared)
    elif adj_shared.is_sparse:
        edge_index = adj_shared.coalesce().indices()
        edge_attr = adj_shared.coalesce().values()
    elif isinstance(adj_shared, np.ndarray):
        adj_shared = torch.tensor(adj_shared, dtype=torch.float)
        edge_index, edge_attr = dense_to_sparse(adj_shared)
    else:
        edge_index, edge_attr = dense_to_sparse(adj_shared)

    node_index = data2input(node_index) if node_index is not None else None
    geo_dataset = Data(edge_index=edge_index, edge_attr=edge_attr, node_index=node_index)
    transform = ToUndirected()
    undirected_geo = transform(geo_dataset)
    return undirected_geo

def data2input(data):
    import torch
    import scipy.sparse as sp
    import numpy as np
    if sp.issparse(data):
        data = data.toarray()
    if not isinstance(data, torch.Tensor):
        data = torch.LongTensor(data) if str(data.dtype).startswith("int") else torch.FloatTensor(data)
    return data

import torch
import numpy as np
from scipy.sparse import coo_matrix

import torch

def sparse_indexing(adj, np_index):
    """
    Indexing a subset of rows and columns from a PyTorch sparse tensor without using for loops.

    Parameters:
        adj (torch.sparse.FloatTensor): Input adjacency matrix in PyTorch sparse format.
        np_index (np.ndarray): Indices of the rows and columns to keep.

    Returns:
        torch.sparse.FloatTensor: Submatrix corresponding to the indices.
    """
    # --- Step 1: Ensure adj is in COO format ---
    adj = adj.coalesce()
    edge_index = adj.indices()
    edge_weight = adj.values()

    # --- Step 2: Create a mask for the edges to keep ---
    row_indices, col_indices = edge_index

    # Convert np_index to a tensor for faster lookups
    np_index_tensor = torch.tensor(np_index, dtype=torch.long)

    # Create the mask using torch.isin for rows and columns
    row_mask = torch.isin(row_indices, np_index_tensor)
    col_mask = torch.isin(col_indices, np_index_tensor)
    mask = row_mask & col_mask

    # --- Step 3: Extract sub-indices and sub-values ---
    sub_row_indices = row_indices[mask]
    sub_col_indices = col_indices[mask]
    sub_values = edge_weight[mask]

    # --- Step 4: Create index map ---
    # Create a tensor that maps old indices to new indices
    index_map = torch.zeros(torch.max(np_index_tensor) + 1, dtype=torch.long)
    index_map[np_index_tensor] = torch.arange(len(np_index_tensor))

    # --- Step 5: Map row and column indices to the new indexing ---
    sub_row_indices = index_map[sub_row_indices]
    sub_col_indices = index_map[sub_col_indices]

    # --- Step 6: Create the sub-adjacency matrix ---
    sub_shape = (len(np_index), len(np_index))
    sub_adj = torch.sparse.FloatTensor(
        torch.stack([sub_row_indices, sub_col_indices]),
        sub_values,
        size=sub_shape
    ).coalesce()

    return sub_adj


########## Data Preprocessing ##########

def clr_normalize_each_cell(adata: anndata.AnnData, inplace: bool = True) -> anndata.AnnData:
    r"""
    Normalize count vector for each cell, i.e., for each row of .X.
    """

    def seurat_clr(x):
        r"""
        CLR normalization function (Seurat implementation).
        """
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    if not inplace:
        adata = adata.copy()

    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.X.A if scipy.sparse.issparse(adata.X) else np.array(adata.X))
    )
    return adata


def lsi(adata: anndata.AnnData, n_components: int = 20, use_highly_variable: bool = None, **kwargs) -> None:
    r"""
    Perform LSI analysis (following the Seurat v3 approach).
    """

    import sklearn.preprocessing
    import sklearn.utils.extmath

    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var

    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata

    X = tfidf(adata_use.X)

    X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
    X_norm = np.log1p(X_norm * 1e4)

    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)

    adata.obsm["X_lsi"] = X_lsi[:, 1:]


def tfidf(X) -> np.ndarray:
    r"""
    Perform TF-IDF normalization (following the Seurat v3 approach).
    """

    import scipy

    idf = X.shape[0] / X.sum(axis=0)

    if scipy.sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf
    
def fix_seed(seed: int, deterministic_cudnn: bool = True, set_hash_seed: bool = True, mode: str = r'') -> None:
    r"""
    Fix random seed for reproducibility.

    Parameters:
        seed (int): Random seed value.
        deterministic_cudnn (bool): Whether to enable deterministic CUDA operations (default: True).
        set_hash_seed (bool): Whether to set the hash seed (default: True).
        mode (str): Mode for strictness (default: r'').
    """
   
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) 
        torch.cuda.manual_seed_all(seed)
        if deterministic_cudnn:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    if set_hash_seed:
        os.environ['PYTHONHASHSEED'] = str(seed)

    # strict mode
    if mode == r'strict':
        ### This will slow down the learning process
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = r':16:8'

def filter_genes_retain(adata, min_cells, retain_genes):
    default_filter = sc.pp.filter_genes(adata, min_cells=min_cells, inplace=False)
    # default_filter = adata.var["n_cells"] >= min_cells

    retain_mask = adata.var_names.isin(retain_genes)
    combined_filter = default_filter | retain_mask

    return adata[:, combined_filter].copy()

import numpy as np
from anndata import AnnData
from typing import Optional, List

def filter_genes_keep_list(
    adata: AnnData,
    gene_list: List[str],
    min_cells: Optional[float] = None,
    copy: bool = False
) -> Optional[AnnData]:
   
    if min_cells is None:
        min_cells_threshold = adata.shape[0] * 0.01  
    else:
        min_cells_threshold = min_cells  

    valid_genes = [gene for gene in gene_list if gene in adata.var_names]
    
    n_cells_per_gene = np.asarray((adata.X > 0).sum(axis=0)).flatten()

    mask_min_cells = n_cells_per_gene >= min_cells_threshold
    mask_keep_genes = adata.var_names.isin(valid_genes)
    final_mask = mask_min_cells | mask_keep_genes

    if copy:
        return adata[:, final_mask].copy()
    else:
        adata._inplace_subset_var(final_mask)

def preprocess_data(adata: anndata.AnnData, modality: str, n_lsi: int = 64, n_top_genes:int=1000, filter_adt: int = 0, gene_list: list = None) -> None:
    r"""
    Preprocess data based on modality.

    Parameters:
        adata (anndata.AnnData): Input AnnData object.
        modality (str): Data modality (e.g., 'rna', 'protein', 'atac').
        n_lsi (int): Number of LSI components (default: 64).
        filter_adt (int): Filter threshold for ADT data (default: 0).
    """
    adata.var_names_make_unique()

    if modality == 'atac':
        sc.pp.filter_genes(adata, min_cells=adata.shape[0] * 0.01)
        lsi(adata, use_highly_variable=False, n_components=n_lsi)

    elif modality == 'rna':
        if gene_list is not None:
            filter_genes_keep_list(adata, gene_list=gene_list, min_cells=0.01)
        else:
            sc.pp.filter_genes(adata, min_cells=adata.shape[0] * 0.01)
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top_genes)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.scale(adata)

        if gene_list is not None:
            highly_var_mask = adata.var['highly_variable']
            marker_mask = adata.var_names.isin(gene_list)
            hvg = highly_var_mask | marker_mask
        else:
            hvg = adata.var['highly_variable']
        adata.var['highly_variable_all'] =hvg
        adata.obsm['X_norm'] = adata[:, hvg].X

    elif modality == 'protein':
        if filter_adt !=0: 
            sc.pp.filter_genes(adata, min_cells=filter_adt)
        adata = clr_normalize_each_cell(adata)
        sc.pp.scale(adata)
        adata.obsm['X_clr'] = adata.X

    elif modality == 'metabolite':
        sc.pp.filter_genes(adata, min_cells=adata.shape[0] * 0.01)
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top_genes)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.scale(adata)
        adata.obsm['X_norm'] = adata[:, adata.var['highly_variable']].X

import torch
from torch_geometric.utils import add_self_loops, degree

def normalize_adj_scr(adj: torch.sparse.FloatTensor, add_loop=True) -> torch.sparse.FloatTensor:
    """
    Normalize adjacency matrix (symmetric normalization: D^{-1/2} A D^{-1/2}).

    Parameters:
        adj (torch.sparse.FloatTensor): Input adjacency matrix in COO sparse format.
        add_loop (bool): Whether to add self-loops to the adjacency matrix. Default is True.

    Returns:
        torch.sparse.FloatTensor: Normalized adjacency matrix in COO sparse format.
    """
    # --- Step 1: Extract edge indices and values from sparse tensor ---
    adj = adj.coalesce()
    edge_index = adj.indices()
    edge_weight = adj.values()
    num_nodes = adj.size(0)

    # --- Step 2: Add self-loops (optional) ---
    if add_loop:
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

    # --- Step 3: Compute degrees and normalization factors ---
    row, col = edge_index
    deg = degree(col, num_nodes=num_nodes, dtype=torch.float32)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0  # Handle isolated nodes

    # --- Step 4: Apply symmetric normalization ---
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    # --- Step 5: Build normalized sparse tensor ---
    normalized_adj = torch.sparse.FloatTensor(
        edge_index,
        norm,
        size=(num_nodes, num_nodes)
    ).coalesce()

    return normalized_adj
 

def normalize_adj(adj_matrix: torch.Tensor) -> torch.sparse.FloatTensor:
    r"""
    Normalize adjacency matrix for dense or sparse tensor format.

    Parameters:
        adj_matrix (torch.Tensor): Input adjacency matrix (dense or sparse tensor).

    Returns:
        torch.sparse.FloatTensor: Normalized adjacency matrix in sparse tensor format.
    """
    num_nodes = adj_matrix.size(0)
    adj_matrix = adj_matrix + torch.eye(num_nodes)
    adj_matrix = (adj_matrix + adj_matrix.t()) / 2
    deg = torch.sum(adj_matrix, dim=1)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj_norm = deg_inv_sqrt.view(-1, 1) * adj_matrix * deg_inv_sqrt.view(1, -1)
    indices = adj_norm.nonzero(as_tuple=False)
    values = adj_norm[indices[:, 0], indices[:, 1]]
    sparse_adj = torch.sparse.FloatTensor(indices.t(), values, adj_matrix.size())
    return sparse_adj


############# Clustering #############

def mclust_R(adata: anndata.AnnData, num_cluster: int, modelNames: str = r'EEE', used_obsm: str = r'emb_pca', random_seed: int = 2024) -> anndata.AnnData:
    r"""
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.

    Parameters:
        adata (anndata.AnnData): Input AnnData object.
        num_cluster (int): Number of clusters.
        modelNames (str): Model name for mclust (default: r'EEE').
        used_obsm (str): Key for the embedding in adata.obsm (default: r'emb_pca').
        random_seed (int): Random seed for reproducibility (default: 2024).

    Returns:
        adata (anndata.AnnData): AnnData object with clustering results stored in adata.obs['mclust'].
    """
    import rpy2.robjects as robjects
    from rpy2.robjects import numpy2ri
    numpy2ri.activate()

    np.random.seed(random_seed)
    robjects.r.library("mclust")
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res.astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata


def pca(adata: anndata.AnnData, use_reps: str = None, n_comps: int = 20) -> np.ndarray:
    r"""
    Dimension reduction with PCA algorithm.

    Parameters:
        adata (anndata.AnnData): Input AnnData object.
        use_reps (str): Key for the input representation in adata.obsm (default: None).
        n_comps (int): Number of PCA components (default: 20).

    Returns:
        feat_pca (numpy.ndarray): PCA-transformed features.
    """
    from sklearn.decomposition import PCA
    from scipy.sparse import csc_matrix, csr_matrix

    pca_model = PCA(n_components=n_comps)
    if use_reps is not None:
        feat_pca = pca_model.fit_transform(adata.obsm[use_reps])
    else:
        if isinstance(adata.X, (csc_matrix, csr_matrix)):
            feat_pca = pca_model.fit_transform(adata.X.toarray())
        else:
            feat_pca = pca_model.fit_transform(adata.X)
    return feat_pca


import torch.nn.functional as F

def norm(array: np.ndarray) -> np.ndarray:
    r"""
    Normalize an array using PyTorch's normalize function.

    Parameters:
        array (np.ndarray): Input array to be normalized.

    Returns:
        np.ndarray: Normalized array.
    """
    import torch
    tensor = torch.tensor(array, dtype=torch.float32)
    normalized_tensor = F.normalize(tensor, p=2, eps=1e-12, dim=1)
    normalized_array = normalized_tensor.detach().numpy()
    return normalized_array


def clustering(adata: anndata.AnnData, n_clusters: int = 7, key: str = r'emb', add_key: str = None, method: str = r'mclust', 
               start: float = 0.05, end: float = 3.0, increment: float = 0.1, use_pca: bool = False, n_comps: int = 20, use_X: bool = False) -> None:
    r"""
    Perform spatial clustering based on latent representations.

    Parameters:
        adata (anndata.AnnData): Input AnnData object containing spatial data.
        n_clusters (int): Target number of clusters (default: 7).
        key (str): Key for accessing embeddings in adata.obsm (default: r'emb').
        add_key (str): Key to store clustering results in adata.obs (default: None).
        method (str): Clustering algorithm (r'mclust', 'leiden', 'louvain', or 'kmeans') (default: r'mclust').
        start (float): Starting resolution for search (leiden/louvain only) (default: 0.05).
        end (float): Maximum resolution for search (leiden/louvain only) (default: 3.0).
        increment (float): Resolution search step size (default: 0.1).
        use_pca (bool): Whether to apply PCA dimensionality reduction (default: False).
        n_comps (int): Number of PCA components if use_pca=True (default: 20).
        use_X (bool): Whether to use adata.X as input features (default: False).
    """
    from sklearn.cluster import KMeans
    import scanpy as sc

    if use_X:
        adata.obsm['X'] = adata.X  
        key = 'X' 

    if use_pca:
        if adata.obsm[key].shape[1] > n_comps:
            adata.obsm[key + '_pca'] = pca(adata, use_reps=key, n_comps=n_comps)
        else:
            use_pca = False  

    if method == 'mclust':
        rep_key = key + '_pca' if use_pca else key
        adata = mclust_R(adata, used_obsm=rep_key, num_cluster=n_clusters)
        if add_key:
            adata.obs[add_key] = adata.obs['mclust']

    elif method == 'leiden':
        rep_key = key + '_pca' if use_pca else key
        res = binary_search_res(
            adata, 
            n_clusters, 
            use_rep=rep_key, 
            method=method, 
            res_min=start,
            res_max=end
        )
        sc.tl.leiden(adata, random_state=2024, resolution=res)
        adata.obs['leiden'] = adata.obs['leiden'].astype('category')
        if add_key:
            adata.obs[add_key] = adata.obs['leiden']

    elif method == 'kmeans':
        rep_key = key + '_pca' if use_pca else key
        kmeans = KMeans(n_clusters=n_clusters, random_state=2024).fit(adata.obsm[rep_key])
        adata.obs['kmeans'] = kmeans.labels_
        adata.obs['kmeans'] = adata.obs['kmeans'].astype('category')
        if add_key:
            adata.obs[add_key] = adata.obs['kmeans']


def binary_search_res(adata: anndata.AnnData, n_clusters: int, method: str = 'leiden', use_rep: str = 'emb', 
                     res_min: float = 0.01, res_max: float = 3.0, max_iter: int = 100, tol: float = 1e-2, 
                     random_state: int = 2024) -> float:
    r"""
    Binary search to find optimal resolution parameter for target cluster number.

    Parameters:
        adata (anndata.AnnData): Input AnnData object.
        n_clusters (int): Target number of clusters.
        method (str): Clustering method ('leiden' or 'louvain').
        use_rep (str): Key for embeddings in adata.obsm.
        res_min (float): Minimum resolution to search.
        res_max (float): Maximum resolution to search.
        max_iter (int): Maximum search iterations.
        tol (float): Tolerance for matching target cluster count.
        random_state (int): Random seed for reproducibility.

    Returns:
        float: Optimal resolution parameter.

    Raises:
        ValueError: If target cluster count cannot be achieved.
    """

    if 'neighbors' not in adata.uns:
        sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)

    for i in range(max_iter):
        current_res = (res_min + res_max) / 2
        print(f"Iteration {i+1}: Testing resolution = {current_res:.4f}")

        if method == 'leiden':
            sc.tl.leiden(adata, resolution=current_res, random_state=random_state)
            n_found = len(adata.obs['leiden'].unique())
        else:
            raise ValueError("Supported methods: 'leiden' or 'louvain'")

        print(f"Resolution {current_res:.4f} produced {n_found} clusters")

        if abs(n_found - n_clusters) <= tol:
            print(f"Optimal resolution found: {current_res:.4f}")
            return current_res
        elif n_found > n_clusters:
            res_max = current_res
        else:
            res_min = current_res

    raise ValueError(f"Failed to find resolution in {max_iter} iterations. " +
                    "Try adjusting res_min/res_max or increasing max_iter.")



################# Evaluation ###################

def super_eval(y_pred, y_true) -> dict:
    r"""
    Evaluates clustering performance using various supervised metrics.
    Handles both numeric and categorical/string labels.

    Args:
        y_pred (array-like): Predicted cluster labels (can be numeric, string or categorical)
        y_true (array-like): True cluster labels (can be numeric, string or categorical)

    Returns:
        dict: Dictionary containing evaluation metrics:
            - AMI: Adjusted Mutual Information
            - NMI: Normalized Mutual Information
            - ARI: Adjusted Rand Index
            - Homogeneity: Homogeneity score
            - V-measure: Harmonic mean of homogeneity and completeness
            - Mutual Information: Raw mutual information

    Raises:
        ValueError: If inputs contain NaN/infinite values or have mismatched lengths
    """
    from sklearn import metrics
    import numpy as np
    import pandas as pd

    def _preprocess_labels(y):
        if isinstance(y, (pd.Categorical, pd.Series)):
            y = y.astype(str) 
        y = np.array(y)
    
        if y.dtype.kind in ['U', 'O']:  
            y = pd.factorize(y)[0]
        return y.astype(float)


    y_pred = _preprocess_labels(y_pred)
    y_true = _preprocess_labels(y_true)

    if len(y_pred) != len(y_true):
        raise ValueError(f"Length mismatch: y_pred ({len(y_pred)}) vs y_true ({len(y_true)})")
    
    if np.isnan(y_true).any():
        raise ValueError("y_true contains NaN values after conversion")
    if np.isnan(y_pred).any():
        raise ValueError("y_pred contains NaN values after conversion")
    if np.isinf(y_true).any():
        raise ValueError("y_true contains infinite values")
    if np.isinf(y_pred).any():
        raise ValueError("y_pred contains infinite values")

    results = {
        'AMI': metrics.adjusted_mutual_info_score(y_true, y_pred),
        'NMI': metrics.normalized_mutual_info_score(y_true, y_pred),
        'ARI': metrics.adjusted_rand_score(y_true, y_pred),
        'Homogeneity': metrics.homogeneity_score(y_true, y_pred),
        'V-measure': metrics.v_measure_score(y_true, y_pred),
        'Mutual Information': metrics.mutual_info_score(y_true, y_pred),
    }

    return results


def unsuper_eval(X, y) -> dict:
    r"""
    Evaluates the performance of a clustering algorithm using various unsupervised metrics.

    Args:
        X (array-like): The feature matrix.
        y (array-like): The predicted cluster labels.

    Returns:
        dict: A dictionary containing the evaluation metrics and their corresponding values.
    """

    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if not isinstance(y, np.ndarray):
        y = np.array(y)

    assert not np.isnan(X).any(), "X contains NaN values"
    assert not np.isnan(y).any(), "y contains NaN values"

    results = {
        'ASW': 0.5 * (silhouette_score(X, y) + 1),
        'DBI': davies_bouldin_score(X, y),
        'CHI': calinski_harabasz_score(X, y),
    }

    return results