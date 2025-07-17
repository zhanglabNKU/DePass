from pathlib import Path
from typing import Optional, Union, Tuple, List
import matplotlib.pyplot as plt
from anndata import AnnData
import scanpy as sc
from typing import Optional


def plot_spatial(  
    adata: AnnData,
    color: str='DePass',  
    save_path: Optional[Union[str, Path]] = None,
    save_name: str = 'spatial_plot',
    title: Optional[str] = None,
    s: int = 35,
    figsize: Tuple[float, float] = (3, 3),
    dpi: int = 300,
    format: str = "pdf",
    frameon: bool = False,
    adjust_margins: bool = True,
    legend_loc: Optional[str] = 'right margin',
    colorbar_loc: Optional[str] = None, 
    show: bool = False,
    **kwargs
) -> None:
    r"""
    Plot spatial data with specified parameters and save the figure.

    Parameters:
        adata (AnnData): Input AnnData object containing spatial data.
        color (str): Column name in adata.obs for coloring the plot.
        save_path (Optional[str]): Path to save the figure (default: None).
        save_name (str): Name of the saved figure.
        title (Optional[str]): Title of the plot (default: None).
        s (int): Size of the markers (default: 35).
        figsize (tuple): Figure size (default: (3, 3)).
        dpi (int): DPI of the saved figure (default: 300).
        format (str): Format of the saved figure (default: "pdf").
        frameon (bool): Whether to show the frame (default: False).
        adjust_margins (bool): Whether to adjust margins (default: True).
        legend_loc (Optional[str]): Location of the legend (default: 'right margin').
        colorbar_loc (Optional[str]): Colorbar position. Set to None to disable colorbar.
        show (bool): Whether to show the plot (default: False).
        **kwargs: Additional keyword arguments for sc.pl.embedding.

    Returns:
        None
    """

    if not isinstance(adata, AnnData):
        raise TypeError("Expected AnnData object, got {}".format(type(adata)))
    
    if not save_name.strip():
        raise ValueError("save_name must contain non-whitespace characters")
    
    file_format = format.lower().lstrip('.')
    allowed_formats = {'png', 'pdf', 'svg', 'tiff', 'jpg', 'jpeg'}
    if file_format not in allowed_formats:
        raise ValueError(f"Invalid format: {format}. Choose from {allowed_formats}")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    if title is None:
        title = color if isinstance(color, str) else ', '.join(color)

    try:
        sc.pl.embedding(
            adata,
            basis='spatial',
            color=color,
            title=title,
            s=s,
            ax=ax,
            show=False,
            frameon=frameon,
            legend_loc=legend_loc,
            colorbar_loc=colorbar_loc, 
            **kwargs
        )
    except KeyError as e:
        raise ValueError(f"Missing required data: {e}") from None

    if save_path is not None:
        output_path = save_path / f"{save_name}.{file_format}"
        try:
            fig.savefig(
                output_path,
                dpi=dpi,
                bbox_inches='tight' if adjust_margins else None,
                pad_inches=0.1 if adjust_margins else 0.5
            )
        except Exception as e:
            raise IOError(f"Failed to save figure: {e}") from None

    if show:
        plt.show()
    
    plt.close(fig)

import pandas as pd
import numpy as np

def getLogFC(
    target_genes: list,
    target_groups: list,
    logfoldchanges: dict,
    gene_names: dict
) -> pd.DataFrame:
    r"""
    Get log fold changes for target genes and groups.

    Parameters:
        target_genes (list): List of target genes.
        target_groups (list): List of target groups.
        logfoldchanges (dict): Dictionary of log fold changes.
        gene_names (dict): Dictionary of gene names.

    Returns:
        pd.DataFrame: DataFrame containing log fold changes for target genes and groups.
    """
    
    if len(target_genes) != len(target_groups):
        raise ValueError("Lengths of `target_genes` and `target_groups` must match.")

    results = []
    for gene, group in zip(target_genes, target_groups):
        # Extract gene names for the current group
        group_genes = gene_names[group]
        # Find index of the target gene in the group's gene list
        gene_idx = np.where(group_genes == gene)[0]
        
        if len(gene_idx) == 0:
            print(f"Warning: Gene '{gene}' not found in group '{group}'.")
            results.append((gene, group, None))  # Store None if gene is missing
        else:
            # Get the first matching LogFC value (assumes unique gene names per group)
            logfc = logfoldchanges[group][gene_idx[0]]
            results.append((gene, group, logfc))
    
    # Convert results to a structured DataFrame
    results_df = pd.DataFrame(results, columns=["Gene", "Group", "LogFC"])
    return results_df


def rank_genes_groups(
    adata,
    groupby: str = "DePass",       
    method: str = "wilcoxon",        
    n_genes: int = 10,             
    standard_scale: str = "var",    
    dpi: int = 300,                
    show: bool = True, 
    save_path: Optional[str] = None,                
    figname: str = 'rank_genes_dotplot',  
) -> None:
    r"""
    Rank genes by groups and optionally plot the results.

    Parameters:
        adata (AnnData): Input AnnData object.
        groupby (str): Column name in adata.obs for grouping (default: "DePass").
        method (str): Method for ranking genes (default: "wilcoxon").
        n_genes (int): Number of top genes to show (default: 10).
        standard_scale (str): Scaling method (default: "var").
        dpi (int): DPI of the saved figure (default: 300).
        show (bool): Whether to show the results (default: True).
        save_path (Optional[str]): Path to save the figure (default: None).
        figname (str): Name of the saved figure (default: 'rank_genes_dotplot').

    Returns:
        None
    """
    
    from sklearn.preprocessing import MinMaxScaler
    import matplotlib.pyplot as plt
     
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

    # Normalize data to [0,1] range for comparative analysis
    scaler = MinMaxScaler()
    adata.obs[groupby] = adata.obs[groupby].astype('str').astype('category')  # Ensure categorical type
    adata.X = scaler.fit_transform(adata.X) 
    
    # Perform differential expression analysis
    sc.tl.rank_genes_groups(adata, groupby=groupby, method=method, use_raw=False)

    if show or save_path is not None:
        sc.pl.rank_genes_groups_dotplot(
            adata,
            groupby=groupby,
            standard_scale=standard_scale,  
            n_genes=n_genes,
            show=False ,
            dendrogram=False,
        )
        if save_path is not None:
           plt.savefig(
               os.path.join(save_path, figname+".pdf"),
               dpi=dpi,
               bbox_inches="tight"  
           )

        if show: plt.show()
        plt.close()

 
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
from sklearn.preprocessing import MinMaxScaler
import os
import warnings

def get_logfc(
    target_gene: str,
    target_group: str,
    logfoldchanges: np.ndarray,  # Structured array from rank_genes_groups
    gene_names: np.ndarray,      # Structured array from rank_genes_groups
) -> float:
   
    # Validate group existence
    if target_group not in gene_names.dtype.names:
        available_groups = list(gene_names.dtype.names)
        raise KeyError(f"Group '{target_group}' not found. Available groups: {available_groups}")
    
    # Locate gene index
    group_genes = gene_names[target_group]
    gene_idx = np.flatnonzero(group_genes == target_gene)
    
    if not gene_idx.size:
        raise KeyError(f"Gene '{target_gene}' not found in group '{target_group}'")
        
    return float(logfoldchanges[target_group][gene_idx[0]])


def plot_marker(
    adata: sc.AnnData,
    target_gene: str,
    save_path: Optional[str] = None,
    save_name: str = "",
    show: bool = True,
    s: int = 80,
    cmap: str = "viridis",
    dpi: int = 300,
    colorbar_loc: Optional[str] = None, 
    figsize: tuple = (3, 3),
    frameon=False,
) -> None:
    r"""
    Plot marker comparison between two datasets.

    Parameters:
        adata (AnnData): AnnData object.
        target_gene (str): Target gene name.
        save_path (Optional[str]): Path to save the figure (default: None).
        save_name (str): Name of the saved figure (default: "gene_comparison").
        show (bool): Whether to show the plot (default: True).
        s (int): Size of the markers (default: 80).
        cmap (str): Colormap (default: "viridis").
        dpi (int): DPI of the saved figure (default: 300).
        colorbar_loc (Optional[str]): Colorbar position. Set to None to disable colorbar.
        figsize (tuple): Figure size (default: (7, 3)).
        frameon (bool): Whether to show the frame (default: False).

    Returns:
        None
    """
    
    if 'spatial' not in adata.obsm:
            raise KeyError(f"Missing spatial coordinates in obsm['spatial']")
    
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
    
    # Shared visualization parameters
    vis_params = {
        'basis': 'spatial',
        'color': f'{target_gene}_expr',
        's': s,
        'frameon': frameon,
        'colorbar_loc': colorbar_loc,
        'cmap': cmap,
    }

    def _scaler_data(adata: sc.AnnData) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            expr = adata[:, target_gene].X.toarray()
        adata.obs[f'{target_gene}_expr'] = MinMaxScaler().fit_transform(expr)

    def _create_plot(adata: sc.AnnData, ax: plt.Axes) -> None:
        sc.pl.embedding(
            adata,
            title=f"{target_gene}",
            ax=ax,
            show=False,
            **vis_params
        )

    
    _scaler_data(adata)
    
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    _create_plot(adata, axes)
    
    if save_path is not None:
        fig.savefig(
            os.path.join(save_path, f"{save_name}.pdf"),
            dpi=dpi,
            bbox_inches="tight"
        )
    if show:
        plt.show()
    plt.close(fig)


def plot_marker_comparison(
    adata1: sc.AnnData,
    adata2: sc.AnnData,
    target_gene: str,
    save_path: Optional[str] = None,
    save_name: str = "gene_comparison",
    show: bool = False,
    s: int = 80,
    cmap: str = "viridis",
    dpi: int = 300,
    colorbar_loc: Optional[str] = None, 
    figsize: tuple = (7, 3),
    frameon=False,
) -> None:
    r"""
    Plot marker comparison between two datasets.

    Parameters:
        adata1 (AnnData): First AnnData object.
        adata2 (AnnData): Second AnnData object.
        target_gene (str): Target gene name.
        save_path (Optional[str]): Path to save the figure (default: None).
        save_name (str): Name of the saved figure (default: "gene_comparison").
        show (bool): Whether to show the plot (default: False).
        s (int): Size of the markers (default: 80).
        cmap (str): Colormap (default: "viridis").
        dpi (int): DPI of the saved figure (default: 300).
        colorbar_loc (Optional[str]): Colorbar position. Set to None to disable colorbar.
        figsize (tuple): Figure size (default: (7, 3)).
        frameon (bool): Whether to show the frame (default: False).

    Returns:
        None
    """
    
    for adata, name in [(adata1, 'adata1'), (adata2, 'adata2')]:
        if 'spatial' not in adata.obsm:
            raise KeyError(f"Missing spatial coordinates in {name}.obsm['spatial']")
    
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
    
    vis_params = {
        'basis': 'spatial',
        'color': f'{target_gene}_expr',
        's': s,
        'frameon':   frameon,
        'colorbar_loc': colorbar_loc,
        'cmap': cmap,
    }

    def _scaler_data(adata: sc.AnnData) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            expr = adata[:, target_gene].X.toarray()
        adata.obs[f'{target_gene}_expr'] = MinMaxScaler().fit_transform(expr)

    def _create_plot(adata: sc.AnnData, ax: plt.Axes,name: str) -> None:
        sc.pl.embedding(
            adata,
            title=f"{name + target_gene}",
            ax=ax,
            show=False,
            **vis_params
        )

    
    _scaler_data(adata1)
    _scaler_data(adata2)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    _create_plot(adata1, axes[0],'Raw - ')
    _create_plot(adata2, axes[1],'Enhanced - ')
    
    if save_path is not None:
        # Save combined comparison
        fig.savefig(
            os.path.join(save_path, f"{save_name}_combined.pdf"),
            dpi=dpi,
            bbox_inches="tight"
        )
    if show:
        plt.show()
    plt.close(fig)


def plot_marker_comparison_with_logFC(
    adata1: sc.AnnData,
    adata2: sc.AnnData,
    target_gene: str,
    target_group: str,
    save_path: Optional[str] = None,
    save_name: str = "gene_comparison",
    show: bool = False,
    s: int = 80,
    cmap: str = "viridis",
    dpi: int = 300,
    colorbar_loc: Optional[str] = None, 
    figsize: tuple = (7, 3),
    frameon=False,
) -> None:
    r"""
    Visualize marker comparison between two datasets with log fold changes.

    Parameters:
        adata1 (AnnData): First AnnData object.
        adata2 (AnnData): Second AnnData object.
        target_gene (str): Target gene name.
        target_group (str): Target group name.
        save_path (Optional[str]): Path to save the figure (default: None).
        save_name (str): Name of the saved figure (default: "gene_comparison").
        show (bool): Whether to show the plot (default: False).
        s (int): Size of the markers (default: 80).
        cmap (str): Colormap (default: "viridis").
        dpi (int): DPI of the saved figure (default: 300).
        colorbar_loc (Optional[str]): Colorbar position. Set to None to disable colorbar.
        figsize (tuple): Figure size (default: (7, 3)).
        frameon (bool): Whether to show the frame (default: False).

    Returns:
        None
    """
    
    for adata, name in [(adata1, 'adata1'), (adata2, 'adata2')]:
        if 'rank_genes_groups' not in adata.uns:
            raise KeyError(f"Missing DEG results in {name}. Run sc.tl.rank_genes_groups first.")
        if 'spatial' not in adata.obsm:
            raise KeyError(f"Missing spatial coordinates in {name}.obsm['spatial']")
   
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
    
    logfc1 = get_logfc(target_gene, target_group,
                               adata1.uns['rank_genes_groups']['logfoldchanges'],
                               adata1.uns['rank_genes_groups']['names'])
    
    logfc2 = get_logfc(target_gene, target_group,
                               adata2.uns['rank_genes_groups']['logfoldchanges'],
                               adata2.uns['rank_genes_groups']['names'])
    
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
    
    # Shared visualization parameters
    vis_params = {
        'basis': 'spatial',
        'color': f'{target_gene}_expr',
        's': s,
        'frameon': frameon,
        'colorbar_loc': colorbar_loc,
        'cmap': cmap,
    }

    def _scaler_data(adata: sc.AnnData) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            expr = adata[:, target_gene].X.toarray()
        adata.obs[f'{target_gene}_expr'] = MinMaxScaler().fit_transform(expr)

    def _create_plot(adata: sc.AnnData, logfc: float, ax: plt.Axes, name:str) -> None:
        sc.pl.embedding(
            adata,
            title=f"{name+target_gene}\n(logFC={logfc:.3f})",
            ax=ax,
            show=False,
            **vis_params
        )

    for adata in [adata1, adata2]:
        _scaler_data(adata)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    _create_plot(adata1, logfc1, axes[0],'Raw - ')
    _create_plot(adata2, logfc2, axes[1],'Enhanced - ')
    
    if save_path is not None:
        fig.savefig(
            os.path.join(save_path, f"{save_name}_combined_logFC.pdf"),
            dpi=dpi,
            bbox_inches="tight"
        )
    if show:
        plt.show()
    plt.close(fig)


#####  li shared the ways of visualized  ####
"""
(1) final_embeddings is a dictionary, and you can input {'s1': embedding_array}.
(2) data_dict is a data dictionary, and you can input:
python
data_dict = {  
    'RNA': [adata1_rna],  
    'Protein': [adata1_adt],  
}  
(3) This function supports K-means clustering by default, so you can specify n_clusters.
(4) If you have already obtained clusters using mclust, set mode='defined' and pass your cluster labels in defined_label (should be an array).
(5) vis_basis refers to the spatial coordinates stored in adata, typically in obsm['spatial']. Note: It is recommended to divide all coordinates 
by 20 to convert them into a grid format (e.g., 0, 1, 2, 3, 4) rather than keeping them at intervals of 20.

"""
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import matplotlib.cm as cm
from matplotlib.colors import to_rgb
import matplotlib.patches as patches

# cluster visualization
def cluster_and_visualize_superpixel(
    final_embeddings,
    data_dict,
    n_clusters,
    mode="joint",  # 'joint' or 'independent'
    defined_labels=None,
    vis_basis="spatial",
    random_state=0,
    colormap=None,
    swap_xy=False,
    invert_x=False,
    invert_y=False,
    offset=False,
    save_path=None,
    dpi=300,
    remove_title = False,
    remove_legend = False,
    remove_spine = False,
    figscale = 35
):
    import numpy as np
    from sklearn.cluster import KMeans
    import os
    import numpy as np


    adata_list = []
    embeddings = []
    coords_all = []
    section_names = []

    for section, embedding in final_embeddings.items():
        idx = int(section[1:]) - 1
        for modality, adata_list_per_mod in data_dict.items():
            if idx < len(adata_list_per_mod) and adata_list_per_mod[idx] is not None:
                adata = adata_list_per_mod[idx]
                adata_list.append(adata)
                embeddings.append(embedding)
                coords = adata.obsm[vis_basis].copy()
                if swap_xy:
                    coords = coords[:, [1, 0]]
                coords = coords.astype(int)
                if offset:
                    offset_value = coords.min(axis=0)     
                    coords -= offset_value               
                coords_all.append(coords)
                section_names.append(section)
                break

    cluster_labels = {}

    if mode == "joint":
        print("Perform joint clustering...")
        combined_embedding = np.vstack(embeddings)
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        all_clusters = kmeans.fit_predict(combined_embedding)
        start = 0
        for section, emb in zip(section_names, embeddings):
            end = start + emb.shape[0]
            cluster_labels[section] = all_clusters[start:end]
            start = end
    elif mode == "independent":
        print("Perform independent clustering...")
        for section, emb in zip(section_names, embeddings):
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
            cluster_labels[section] = kmeans.fit_predict(emb)
    elif mode == 'defined':
        if defined_labels is None:
            raise ValueError("If mode='defined', you must provide `defined_labels`.")
        cluster_labels = defined_labels
    else:
        raise ValueError("mode must be 'joint' or 'independent'")

    for section, coords, labels in zip(section_names, coords_all, cluster_labels.values()):
        max_y, max_x = coords.max(axis=0) + 1
        image = np.full((max_y, max_x), fill_value=-1, dtype=int)
        for (y, x), label in zip(coords, labels):
            image[y, x] = label
        if invert_x:
            image = image[:, ::-1]
        if invert_y:
            image = image[::-1, :]
        section_save_path = None
        if save_path:
            base, ext = os.path.splitext(save_path)
            section_save_path = f"{base}_section_{section}{ext or '.png'}"
        
        plot_histology_clusters(
            he_clusters_image=image,
            num_he_clusters=n_clusters,
            section_title=f"Section {section} ({mode})",
            colormap=colormap,
            save_path=section_save_path,
            dpi=dpi,
            figscale = figscale,
            remove_title = remove_title,
            remove_legend = remove_legend,
            remove_spine=remove_legend, 
        )

    return cluster_labels



def plot_histology_clusters(he_clusters_image,
                            num_he_clusters,
                            section_title=None,
                            colormap=None,
                            save_path=None,
                            figscale = 35,
                            remove_title = False,
                            remove_legend = False,
                            remove_spine=False, 
                            dpi=300):


    if colormap is None:
        color_list = [[255,127,14],[44,160,44],[214,39,40],[148,103,189],
                      [140,86,75],[227,119,194],[127,127,127],[188,189,34],
                      [23,190,207],[174,199,232],[255,187,120],[152,223,138],
                      [255,152,150],[197,176,213],[196,156,148],[247,182,210],
                      [199,199,199],[219,219,141],[158,218,229],[16,60,90],
                      [128,64,7],[22,80,22],[107,20,20],[74,52,94],[70,43,38],
                      [114,60,97],[64,64,64],[94,94,17],[12,95,104],[0,0,0]]

    elif isinstance(colormap, list):
        color_list = colormap

    else:
        cmap = cm.get_cmap(colormap)
        color_list = [ [int(255 * c) for c in to_rgb(cmap(i))] for i in range(len(cmap.colors)) ]

    image_rgb = 255 * np.ones([he_clusters_image.shape[0], he_clusters_image.shape[1], 3])
    for cluster in range(num_he_clusters):
        image_rgb[he_clusters_image == cluster] = color_list[cluster]
    image_rgb = np.array(image_rgb, dtype='uint8')

    plt.figure(figsize=(he_clusters_image.shape[1] // figscale, he_clusters_image.shape[0] // figscale))
    if remove_title:
        plt.title("")
    else:
        title = section_title if section_title else "Histology Clusters"
        plt.title(title, fontsize=18)
    plt.imshow(image_rgb, interpolation='none')
    # plt.show()
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])

    if remove_spine:
        for spine in ax.spines.values():
            spine.set_visible(False)

    if not remove_legend:
        legend_elements = [patches.Patch(facecolor=np.array(color_list[i]) / 255,
                                         label=f'Cluster {i}')
                           for i in range(num_he_clusters)]
        plt.legend(handles=legend_elements,
                   bbox_to_anchor=(1.05, 1),
                   loc='upper left',
                   borderaxespad=0.,
                   fontsize=12)

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()
    # plt.close()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.colors import to_rgb
import matplotlib.patches as patches
import os


def plot_superpixel(
    adata,
    label_key='label',  
    vis_basis='spatial',  
    colormap=None,
    save_path=None,
    save_name='cluster_visualization',
    title=None,
    figscale=100,
    format='png',
    show=True,
    remove_title=False,
    remove_legend=False,
    remove_spine=False,
    dpi=300,
    random_state=2024,  
    swap_xy=False,  
    invert_x=False,  
    invert_y=False  
):
    """
    Visualize clusters using labels and spatial coordinates from adata.

    Parameters:
        adata: AnnData object containing cell data and cluster labels.
        label_key: Key for cluster labels in adata.obs.
        vis_basis: Key for coordinates in adata.obsm.
        colormap: Custom color mapping.
        save_path: Directory path to save the image.
        save_name: File name to save the image.
        title: Title of the plot.
        figscale: Figure size scaling factor.
        format: Format to save the image (e.g., 'pdf', 'png').
        show: Whether to display the plot.
        remove_title: Whether to remove the title.
        remove_legend: Whether to remove the legend.
        remove_spine: Whether to remove the border.
        dpi: Image resolution.
        random_state: Random seed for reproducibility.
        swap_xy: Whether to swap x and y coordinates.
        invert_x: Whether to invert x-axis.
        invert_y: Whether to invert y-axis.
    """
   
    np.random.seed(random_state)

   
    labels = adata.obs[label_key].values.astype(int)
    coords = adata.obsm[vis_basis].copy().astype(int)

    if swap_xy:
        coords = coords[:, [1, 0]]

    unique_labels = np.unique(labels)
    num_clusters = len(unique_labels)

    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    zero_based_labels = np.array([label_to_index[label] for label in labels])


    if colormap is None:
        color_list = [
    [60, 142, 204],    
    [187, 187, 187],   
    [246, 216, 208],   
    [254, 238, 237],   
    [215, 102, 102],   
    [177, 157, 177],   
    [60, 162, 254],    
    [151, 215, 243],   
    [208, 163, 239],   
    [246, 216, 212],   
    [255, 247, 180],   
    [241, 91, 108],    
    [60, 188, 60],     
    [104, 220, 104],   
    [247, 172, 188],   
    [222, 171, 138],   
    [255, 188, 188],   
    [199, 133, 89],    
    [60, 251, 255],    
    [195, 236, 255],   
    [204, 238, 204],   
    [254, 220, 189],   
    [239, 91, 156],    
    [176,224,230],   
    [187, 187, 187]    
]
    elif isinstance(colormap, list):
        color_list = colormap
    else:
        cmap = cm.get_cmap(colormap)
        color_list = [[int(255 * c) for c in to_rgb(cmap(i))] for i in range(len(cmap.colors))]

    if len(color_list) < num_clusters:
        raise ValueError("Color list is not long enough to cover all clusters.")

    max_y, max_x = coords.max(axis=0) + 1
    image = np.full((max_y, max_x), fill_value=-1, dtype=int)
    for (y, x), label in zip(coords, zero_based_labels):
        if 0 <= x < max_x and 0 <= y < max_y:
            image[y, x] = label

    if invert_x:
        image = image[:, ::-1]
    if invert_y:
        image = image[::-1, :]

    image_rgb = 255 * np.ones([image.shape[0], image.shape[1], 3])
    for cluster in range(num_clusters):
        image_rgb[image == cluster] = color_list[cluster]
    image_rgb = np.array(image_rgb, dtype='uint8')

    
    plt.figure(figsize=(image.shape[1] // figscale, image.shape[0] // figscale))
    if remove_title or title is None:
        plt.title("")
    else:
        plt.title(title, fontsize=18)
    plt.imshow(image_rgb, interpolation='none')
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])

    if remove_spine:
        for spine in ax.spines.values():
            spine.set_visible(False)

    if not remove_legend:
        legend_elements = [patches.Patch(facecolor=np.array(color_list[i]) / 255,
                                          label=f'Cluster {unique_labels[i]}') for i in range(num_clusters)]
        plt.legend(handles=legend_elements,
                   bbox_to_anchor=(1.05, 1),
                   loc='upper left',
                   borderaxespad=0.,
                   fontsize=12)

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(save_path, f"{save_name}.{format}")
        plt.savefig(file_path, dpi=dpi, bbox_inches="tight", format=format)
        print(f"Image saved to: {file_path}")

    if show:
        plt.show()
    else:
        plt.close()


# marker visualization
def prepare_image(adata, molecule_name, basis, swap_xy, invert_x, invert_y, offset, scale):
    coords = adata.obsm[basis].copy()
    if swap_xy:
        coords = coords[:, [1, 0]]
    coords = coords.astype(int)
    if offset:
        offset_value = coords.min(axis=0)
        coords -= offset_value 

    values = adata[:, molecule_name].X
  

    if hasattr(values, "toarray"):
        values = values.toarray().flatten()
    else:
        values = np.array(values).flatten()

    if scale:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        values = values.reshape(-1, 1)  
        values = scaler.fit_transform(values)
        values = values.flatten()  
  

    max_y, max_x = coords.max(axis=0) + 1
    image = np.full((max_y, max_x), np.nan, dtype=float)
    for (y, x), val in zip(coords, values):
        image[y, x] = val

    if invert_x:
        image = image[:, ::-1]
    if invert_y:
        image = image[::-1, :]

    return image

def plot_marker_comparison_superpixel(
    molecule_name: str,
    adata1,
    adata2,
    section1_label: str = 'Section 1',
    section2_label: str = 'Section 2',
    basis: str = 'spatial',
    colormap: str = "viridis",
    plot_style: str = "original",
    scale: bool = True,
    swap_xy: bool = False,
    invert_x: bool = False,
    invert_y: bool = False,
    offset: bool = False,
    figscale: int = 35,
    dpi: int = 300,
    remove_title: bool = False,     
    remove_spine: bool = False,    
    remove_legend: bool = False,      
    save_path: str = None
):


    img1 = prepare_image(adata1, molecule_name, basis, swap_xy, invert_x, invert_y, offset, scale)
    img2 = prepare_image(adata2, molecule_name, basis, swap_xy, invert_x, invert_y, offset, scale)


    figsize1 = (img1.shape[1] / figscale, img1.shape[0] / figscale)
    figsize2 = (img2.shape[1] / figscale, img2.shape[0] / figscale)
    figsize = (figsize1[0] + figsize2[0], max(figsize1[1], figsize2[1]))

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, img, title in zip(axes, [img1, img2], [section1_label, section2_label]):
        im = ax.imshow(img, cmap=colormap, interpolation='none')
        if not remove_title:
            ax.set_title(f"{title} - {molecule_name}", fontsize=16)
        else:
            ax.set_title("")
        ax.set_xticks([])
        ax.set_yticks([])
        if remove_spine:
            for spine in ax.spines.values():
                spine.set_visible(False)
        if plot_style == "equal":
            ax.set_aspect("equal")

        if not remove_legend:
            cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)  

    if save_path:
        base, ext = os.path.splitext(save_path)
        if not ext:
            ext = ".png"
        save_path = base + ext
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"Saving marker comparison to: {save_path}")
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    plt.show()
    plt.close()


import pandas as pd
from typing import List, Dict, Union
from pathlib import Path
from anndata import AnnData


def get_logfc_df(
    adata_list: List[AnnData],
    adata_names: List[str],
    target_genes: List[str],
    target_groups: List[str],
    save_path: Union[str, Path] = "results",
    save_name: str = "logfc_comparison"
) -> pd.DataFrame:
    r"""
    Get log fold change DataFrame for multiple datasets.

    Parameters:
        adata_list (List[AnnData]): List of AnnData objects.
        adata_names (List[str]): List of dataset names.
        target_genes (List[str]): List of target genes.
        target_groups (List[str]): List of target groups.
        save_path (Union[str, Path]): Path to save the DataFrame (default: "results").
        save_name (str): Name of the saved file (default: "logfc_comparison").

    Returns:
        pd.DataFrame: DataFrame containing log fold changes.
    """
    
    if len(target_genes) != len(target_groups):
        raise ValueError("Length of target_genes and target_groups must match!")
    
    if len(adata_list) != len(adata_names):
        raise ValueError("Length of adata_list and adata_names must match!")
    
    results = {
        'Gene': target_genes,
        'Group': target_groups
    }
    
    # Extract logFC values for each dataset
    for adata, name in zip(adata_list, adata_names):
        logfc_values = []
        for gene, group in zip(target_genes, target_groups):
            try:
                logfc = get_logfc(
                    target_gene=gene,
                    target_group=group,
                    logfoldchanges=adata.uns['rank_genes_groups']['logfoldchanges'],
                    gene_names=adata.uns['rank_genes_groups']['names']
                )
                logfc_values.append(logfc)
            except KeyError as e:
                print(f"[Warning] Failed to retrieve {gene}@{group} in dataset {name}: {str(e)}")
                logfc_values.append(None)
        
        results[f'logFC_{name}'] = logfc_values
    
    # Convert to DataFrame
    logfc_df = pd.DataFrame(results)
    
    # Melt the DataFrame for visualization
    value_vars = [col for col in logfc_df.columns if col.startswith('logFC_')]
    long_df = pd.melt(
        logfc_df,
        id_vars=['Gene', 'Group'],
        value_vars=value_vars,
        var_name='type',
        value_name='logFC'
    )
    
    long_df['type'] = long_df['type'].str.replace('logFC_', '')
    
    output_path = Path(save_path)
    output_path.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(output_path / f"{save_name}.csv", sep='\t', index=False)
    
    return long_df

import pandas as pd
from anndata import AnnData
from typing import Optional, Union

def get_top_degs_df(
    adata: AnnData,
    n_top_genes: int = 20,
    groupby: Optional[str] = None,
    key: str = 'rank_genes_groups'
) -> pd.DataFrame:
    r"""
    Extract top N differentially expressed genes (DEGs) and their statistics from Scanpy's rank_genes_groups results.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing rank_genes_groups results.
    n_top_genes : int, optional
        Number of top DEGs to extract per group (default: 20).
    groupby : str, optional
        Column name in adata.obs used for grouping. If None, automatically reads from rank_genes_groups.
    key : str, optional
        Key name for rank_genes_groups in adata.uns (default: 'rank_genes_groups').
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing the following columns:
        - Group: Group name
        - Gene: Gene symbol
        - LogFC: Log fold change
        - PValue: Raw p-value
        - AdjPValue: Adjusted p-value (e.g., FDR)
    """

    if key not in adata.uns:
        raise KeyError(f"'{key}' not found in adata.uns. Run sc.tl.rank_genes_groups first.")
    
    rank_data = adata.uns[key]
    required_fields = ['names', 'logfoldchanges', 'pvals', 'pvals_adj']
    for field in required_fields:
        if field not in rank_data:
            raise ValueError(f"Missing required field '{field}' in rank_genes_groups data.")

    if groupby is None:
        groupby = rank_data['params']['groupby'] if 'params' in rank_data else None
        if groupby is None:
            raise ValueError("Please specify them manually through the 'groupby' parameter.")


    groups = rank_data['names'].dtype.names
    gene_names = rank_data['names']
    logfcs = rank_data['logfoldchanges']
    pvals = rank_data['pvals']
    pvals_adj = rank_data['pvals_adj']

    top_genes = []
    for group in groups:
        genes = gene_names[group][:n_top_genes]
        valid_idx = ~pd.isnull(genes)
        genes = genes[valid_idx]
        
        group_logfcs = logfcs[group][:n_top_genes][valid_idx]
        group_pvals = pvals[group][:n_top_genes][valid_idx]
        group_padjs = pvals_adj[group][:n_top_genes][valid_idx]
        
        for gene, lfc, pval, padj in zip(genes, group_logfcs, group_pvals, group_padjs):
            top_genes.append({
                'Group': group,
                'Gene': gene,
                'LogFC': lfc,
                'PValue': pval,
                'AdjPValue': padj
            })

    return pd.DataFrame(top_genes)

from typing import Optional, Dict
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from anndata import AnnData

def plot_modality_weights(
    adata: AnnData,
    modality_names: tuple = ("RNA", "Protein"),
    cluster_column: str = "DePass",
    save_path: Optional[str] = None,
    save_name: str = "modality_weights",
    show: bool = True,
    figsize: tuple = (5, 3),
    palette: Dict[str, str] = None,
    **kwargs
) -> plt.Axes:
    r"""
    Plot modality weights for clusters with legend outside right.
    """
 
    if 'alpha' not in adata.obsm:
        raise KeyError("Missing modality weights in adata.obsm['alpha']")
    if cluster_column not in adata.obs:
        raise KeyError(f"Cluster column '{cluster_column}' not found in adata.obs")
    if adata.obsm['alpha'].shape[1] != 2:
        raise ValueError(f"Expected 2 columns in adata.obsm['alpha'], found {adata.obsm['alpha'].shape[1]}")

    default_palette = {modality_names[0]: "#CAB82E", modality_names[1]: "#9368A6"}
    palette = palette or default_palette

    plot_df = pd.DataFrame({
        modality_names[0]: adata.obsm['alpha'][:, 0],
        modality_names[1]: adata.obsm['alpha'][:, 1],
        'Cluster': adata.obs[cluster_column].astype(str)
    })
    melted_df = plot_df.melt(
        id_vars='Cluster',
        value_vars=modality_names,
        var_name='Modality',
        value_name='Weight'
    )
    clusters = sorted(plot_df['Cluster'].unique(), key=lambda x: int(x))
    melted_df['Cluster'] = pd.Categorical(melted_df['Cluster'], categories=clusters, ordered=True)

    plt.figure(figsize=figsize)
    ax = sns.violinplot(
        data=melted_df,
        x='Cluster',
        y='Weight',
        hue='Modality',
        inner="quart",
        linewidth=0.5,
        palette=palette,
        **kwargs
    )

    ax.set_title(f"{modality_names[0]} vs {modality_names[1]}", pad=15)
    ax.set_xlabel("Cluster", labelpad=10)
    ax.set_ylabel("Attention Weight", labelpad=10)

    ax.legend(
        loc='center left',
        bbox_to_anchor=(1.05, 0.5),
        frameon=True,
        title='Modality'
    )
    
    plt.subplots_adjust(right=0.75)
    plt.tight_layout()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(
            os.path.join(save_path, f"{save_name}.pdf"),
            dpi=300,
            bbox_inches="tight",
            transparent=True
        )
    if show:
        plt.show()
    plt.close()

    return ax


def noise(adata_omics1, adata_omics2, level):
    r"""
    Add noise to omics data.

    Parameters:
        adata_omics1 (AnnData): First AnnData object.
        adata_omics2 (AnnData): Second AnnData object.
        level (float): Noise level.

    Returns:
        tuple: Noisy omics data.
    """
    from scipy import stats
    noise1 = stats.norm.rvs(loc=0, scale=level, size=adata_omics1.obsm['X_norm'].shape)
    noise2 = stats.norm.rvs(loc=0, scale=level, size=adata_omics2.obsm['X_clr'].shape)

    return adata_omics1.obsm['X_norm'] + noise1, adata_omics2.obsm['X_clr'] + noise2


def Generate_masked_data(adata_omics1,adata_omics2,mask_ratio):
    r"""
    Generate masked data.

    Parameters:
        mask_ratio (float): Masking ratio.

    Returns:
        tuple: Masked data.
    """
    import numpy as np
    from scipy import sparse
    import scanpy as sc
    from DePass.utils import preprocess_data

    adata_omics1.var_names_make_unique()
    adata_omics2.var_names_make_unique()

    def mask(adata, ratio):
        X = adata.X.copy()  

        if sparse.issparse(X):
            nonzero_positions = X.nonzero()
        else:
            nonzero_positions = np.nonzero(X)

        num_to_mask = int(ratio * len(nonzero_positions[0]))
        mask_indices = np.random.choice(len(nonzero_positions[0]), num_to_mask, replace=False)
        
        if sparse.issparse(X):
            X[(nonzero_positions[0][mask_indices], nonzero_positions[1][mask_indices])] = 0
        else:
            X[nonzero_positions[0][mask_indices], nonzero_positions[1][mask_indices]] = 0

        adata.X = X
        return adata

    sc.pp.filter_genes(adata_omics1, min_cells=adata_omics1.shape[0] * 0.01)
    sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=1000)
    adata_omics1 = adata_omics1[:, adata_omics1.var['highly_variable']]
    adata_omics1 = mask(adata_omics1, mask_ratio)
    sc.pp.normalize_total(adata_omics1, target_sum=1e4)
    sc.pp.log1p(adata_omics1)
    sc.pp.scale(adata_omics1)

    adata_omics2 = mask(adata_omics2, mask_ratio)
    preprocess_data(adata_omics2, modality='protein')
    return adata_omics1.X, adata_omics2.X


import scanpy as sc
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_correlation(adata1, adata2, gene_adt_mapping):
    r"""
    Calculate correlations between genes and ADTs.

    Parameters:
        adata1 (AnnData): First AnnData object (genes).
        adata2 (AnnData): Second AnnData object (ADTs).
        gene_adt_mapping (dict): Mapping from genes to ADTs.

    Returns:
        pd.DataFrame: DataFrame containing correlation results.
    """
    results = []
    for adt, genes in gene_adt_mapping.items():
        if adt not in adata2.var_names:
            print(f"Warning: {adt} not found in ADT data.")
            continue
        
        for gene in genes:
            if gene not in adata1.var_names:
                print(f"Warning: {gene} not found in gene data.")
                continue
            
            # Extract expression values
            gene_expression = adata1[:, gene].X.flatten()  # Gene expression
            adt_expression = adata2[:, adt].X.flatten()    # ADT expression
            
            # Calculate Pearson correlation coefficient
            correlation, p_value = pearsonr(gene_expression, adt_expression)
            
            results.append({
                "ADT": adt,
                "Gene": gene,
                "Gene_ADT": f"{gene}_{adt}",  
                "Correlation": correlation,
                "P_value": p_value
            })
    
    return pd.DataFrame(results)