import scanpy as sc
import numpy as np
from anndata import AnnData
import os

save_dir = 'outputs' 
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

adata = sc.read('/home/jyx2/DePass-main/outputs/dataset_Spatial_Mux_seq/run/adata_raw.h5ad')
# adata = sc.read('/home/jyx2/DePass-main/outputs/dataset_Spatial_Mux_seq/run/adata_enhanced.h5ad')


"""
We evaluated DePass on the mouse embryo dataset by performing pseudotime analysis 
using both raw RNA data and DePass-enhanced RNA data, focusing on the radial glia (cluster 4) 
and spinal cord (cluster 3) region. 
"""

mask = adata.obs['DePass'].isin([3, 4])
adata_sub = AnnData(
    X=adata.X[mask].copy(),
    obs=adata.obs[mask].copy()
)
adata_sub.obsm['spatial'] = adata.obsm['spatial'][mask].copy()
cluster4_cells = adata_sub.obs.index[adata_sub.obs['DePass'] == 4]
np.random.seed(10)
first_spot_index = np.random.choice(cluster4_cells)
first_spot_row_index = adata_sub.obs.index.get_loc(first_spot_index)

sc.pp.neighbors(adata_sub, n_neighbors=30)
sc.tl.diffmap(adata_sub)
adata_sub.uns['iroot'] = first_spot_row_index
sc.tl.dpt(adata_sub)
adata.obs['dpt_pseudotime_34'] = np.nan
adata.obs.loc[
    adata_sub.obs.index,
    'dpt_pseudotime_34'
] = adata_sub.obs['dpt_pseudotime']

adata.write(save_dir+'/adata_raw.h5ad')


