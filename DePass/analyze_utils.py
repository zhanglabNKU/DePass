from pathlib import Path
from typing import Optional, Union, Tuple, List
import matplotlib.pyplot as plt
from anndata import AnnData
import scanpy as sc
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import seaborn as sns
import warnings


def plot_spatial(  
    adata: AnnData,
    color: str = 'DePass',  
    save_path: Optional[Union[str, Path]] = None,
    save_name: str = 'spatial_plot',
    title: Optional[str] = None,
    s: int = 35,
    figsize: Tuple[float, float] = (3, 3),
    dpi: int = 300,
    format: str = "png",
    frameon: bool = True,
    adjust_margins: bool = True,
    legend_loc: Optional[str] = 'right margin',
    colorbar_loc: Optional[str] = 'right', 
    show: bool = False,
    **kwargs
) -> None:
    """
    Plot spatial data using `scanpy.pl.embedding`.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing spatial coordinates in `adata.obsm['spatial']`.
    color : str, default="DePass"
        Column name in `adata.obs` or gene name to color the plot.
    save_path : str or Path, optional
        Directory where the figure will be saved. If None, the figure is not saved.
    save_name : str, default="spatial_plot"
        Filename (without extension) for saving the plot.
    title : str, optional
        Title of the plot. If None, defaults to the `color` argument.
    s : int, default=35
        Marker size.
    figsize : tuple of float, default=(3, 3)
        Figure size in inches.
    dpi : int, default=300
        Resolution of the saved figure.
    format : {"png", "pdf", "svg", "tiff", "jpg", "jpeg"}, default="png"
        Output file format.
    frameon : bool, default=True
        Whether to show a frame around the plot.
    adjust_margins : bool, default=True
        Whether to tighten layout by adjusting margins.
    legend_loc : str or None, default="right margin"
        Position of the legend. Set to None to disable.
    colorbar_loc : str or None, default="right"
        Position of the colorbar. Set to None to disable.
    show : bool, default=False
        Whether to display the plot interactively.
    **kwargs
        Additional arguments passed to `scanpy.pl.embedding`.

    Returns
    -------
    None
        The function generates a spatial plot and optionally saves it.
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
            plt.gca().set_rasterized(True)  
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


def getLogFC(
    target_genes: list,
    target_groups: list,
    logfoldchanges: dict,
    gene_names: dict
) -> pd.DataFrame:
    """
    Extract log fold changes (LogFC) for a list of target genes in specific groups.

    Parameters
    ----------
    target_genes : list of str
        List of gene names of interest.
    target_groups : list of str
        List of groups corresponding to each target gene.
    logfoldchanges : dict
        Dictionary mapping each group to an array of log fold change values.
    gene_names : dict
        Dictionary mapping each group to an array of gene names.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - "Gene": Gene name
        - "Group": Group name
        - "LogFC": Log fold change value (or None if not found)

    Raises
    ------
    ValueError
        If the lengths of `target_genes` and `target_groups` do not match.
    """

    if len(target_genes) != len(target_groups):
        raise ValueError("Lengths of `target_genes` and `target_groups` must match.")

    results = []
    for gene, group in zip(target_genes, target_groups):
        group_genes = gene_names[group]
        gene_idx = np.where(group_genes == gene)[0]
        
        if len(gene_idx) == 0:
            print(f"Warning: Gene '{gene}' not found in group '{group}'.")
            results.append((gene, group, None))  
        else:
            logfc = logfoldchanges[group][gene_idx[0]]
            results.append((gene, group, logfc))
    
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
    figsize: Tuple[float, float] = (6, 3),
) -> None:
    """
    Perform differential expression analysis and visualize ranked genes.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    groupby : str, default="DePass"
        Column in `adata.obs` used for grouping cells.
    method : str, default="wilcoxon"
        Statistical test method. Options are supported by Scanpy.
    n_genes : int, default=10
        Number of top genes to display per group.
    standard_scale : {"var", "group"}, default="var"
        Whether to standardize by variable or group.
    dpi : int, default=300
        Resolution of the output figure.
    show : bool, default=True
        Whether to display the plot.
    save_path : str, optional
        Directory to save the plot. If None, the plot is not saved.
    figname : str, default="rank_genes_dotplot"
        Filename (without extension) for saving.
    figsize : tuple of float, default=(6, 3)
        Figure size in inches.

    Returns
    -------
    None
        The function runs DE analysis, produces a dot plot, and optionally saves it.
    """
    
    from sklearn.preprocessing import MinMaxScaler

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

    # Normalize data to [0,1] range for comparative analysis
    scaler = MinMaxScaler()
    adata.obs[groupby] = adata.obs[groupby].astype('str').astype('category')  # Ensure categorical type
    adata.X = scaler.fit_transform(adata.X) 
    

    sc.tl.rank_genes_groups(adata, groupby=groupby, method=method, use_raw=False)

    if show or save_path is not None:
        sc.pl.rank_genes_groups_dotplot(
            adata,
            groupby=groupby,
            standard_scale=standard_scale,  
            n_genes=n_genes,
            show=False ,
            dendrogram=False,
            figsize=figsize,
        )
        if save_path is not None:
           plt.savefig(
               os.path.join(save_path, figname+".png"),
               dpi=dpi,
               bbox_inches="tight"  
           )

        if show: plt.show()
        plt.close()

 


def get_logfc(
    target_gene: str,
    target_group: str,
    logfoldchanges: np.ndarray,  
    gene_names: np.ndarray,     
) -> float:
    """
    Retrieve log fold change for a specific gene in a target group.

    Parameters
    ----------
    target_gene : str
        Gene of interest.
    target_group : str
        Group of interest.
    logfoldchanges : np.ndarray
        Structured array of log fold changes from `rank_genes_groups`.
    gene_names : np.ndarray
        Structured array of gene names from `rank_genes_groups`.

    Returns
    -------
    float
        Log fold change value for the specified gene in the target group.

    Raises
    ------
    KeyError
        If the group or gene is not found.
    """
   
    if target_group not in gene_names.dtype.names:
        available_groups = list(gene_names.dtype.names)
        raise KeyError(f"Group '{target_group}' not found. Available groups: {available_groups}")
    
    group_genes = gene_names[target_group]
    gene_idx = np.flatnonzero(group_genes == target_gene)
    
    if not gene_idx.size:
        raise KeyError(f"Gene '{target_gene}' not found in group '{target_group}'")
        
    return float(logfoldchanges[target_group][gene_idx[0]])

def plot_marker_comparison(
    adata1: sc.AnnData,
    adata2: sc.AnnData,
    target_gene: str,
    save_path: Optional[str] = None,
    save_name: str = "gene_comparison",
    show: bool = False,
    s: int = 80,
    cmap: str = "turbo",
    dpi: int = 300,
    colorbar_loc: Optional[str] = None, 
    figsize: tuple = (7, 3),
    frameon=False,
) -> None:
    """
    Plot expression of selected marker genes across groups using a dot plot.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    markers : list of str
        List of marker genes to plot.
    groupby : str, default="DePass"
        Column in `adata.obs` used for grouping cells.
    dpi : int, default=300
        Resolution of the figure.
    show : bool, default=True
        Whether to display the plot.
    save_path : str, optional
        Directory where the figure will be saved. If None, the figure is not saved.
    figname : str, default="marker_comparison_dotplot"
        Filename (without extension) for saving.
    figsize : tuple of float, default=(6, 3)
        Figure size in inches.

    Returns
    -------
    None
        The function generates a dot plot and optionally saves it.
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
        try:
            plt.gca().set_rasterized(True)  
            fig.savefig(
            os.path.join(save_path, f"{save_name}_combined.png"),
            dpi=dpi,
            bbox_inches="tight"
               )
        except Exception as e:
            raise IOError(f"Failed to save figure: {e}") from None

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
    cmap: str = "turbo",
    dpi: int = 300,
    colorbar_loc: Optional[str] = None, 
    figsize: tuple = (7, 3),
    frameon=False,
) -> None:
    """
    Compare spatial expression of a target gene between two datasets,
    displaying log fold change (logFC) values from differential expression results.

    The function extracts logFC for the given `target_gene` in the specified 
    `target_group` from both datasets, rescales expression values for visualization, 
    and plots spatial embeddings side-by-side (e.g., raw vs enhanced).

    Parameters
    ----------
    adata1 : AnnData
        First annotated data matrix (e.g., raw data). Must contain:
        - `adata1.uns['rank_genes_groups']` with DE results
        - `adata1.obsm['spatial']` with spatial coordinates.
    adata2 : AnnData
        Second annotated data matrix (e.g., enhanced data). Same requirements as `adata1`.
    target_gene : str
        Gene of interest to visualize.
    target_group : str
        Group/cluster in which the logFC of `target_gene` is extracted.
    save_path : str, optional
        Directory to save the figure. If None, the figure is not saved.
    save_name : str, default="gene_comparison"
        Filename (without extension) for saving.
    show : bool, default=False
        Whether to display the plots interactively.
    s : int, default=80
        Dot size for the scatter plot.
    cmap : str, default="turbo"
        Colormap used for gene expression visualization.
    dpi : int, default=300
        Resolution of the saved figure.
    colorbar_loc : str, optional
        Location of the colorbar. If None, no colorbar is shown.
    figsize : tuple of float, default=(7, 3)
        Figure size in inches.
    frameon : bool, default=False
        Whether to draw a frame around the embedding.

    Returns
    -------
    None
        The function generates side-by-side spatial plots of the target gene in both datasets,
        annotated with logFC values, and optionally saves them.
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
            title=f"{name+target_gene}\n(LogFC={logfc:.3f})",
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
        plt.gca().set_rasterized(True)
        fig.savefig(
            os.path.join(save_path, f"{save_name}_combined_logFC.png"),
            dpi=dpi,
            bbox_inches="tight"
        )
    if show:
        plt.show()
    plt.close(fig)




import matplotlib.cm as cm
from matplotlib.colors import to_rgb
import matplotlib.patches as patches


def cluster_and_visualize_superpixel(
    final_embeddings,
    data_dict,
    n_clusters,
    mode="joint",  
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
    """
    Cluster superpixel embeddings and visualize results on histology sections.

    This function takes precomputed embeddings of superpixels (`final_embeddings`), 
    aligns them with corresponding spatial coordinates from `data_dict`, performs clustering, 
    and visualizes each section as a cluster-labeled image.

    Parameters
    ----------
    final_embeddings : dict of {str: np.ndarray}
        Dictionary mapping section IDs (e.g., "S1", "S2") to superpixel embeddings.
    data_dict : dict of {str: list of AnnData or None}
        Dictionary mapping modality names to lists of AnnData objects per section.
        Used to extract spatial coordinates (`adata.obsm[vis_basis]`).
    n_clusters : int
        Number of clusters for KMeans.
    mode : {"joint", "independent", "defined"}, default="joint"
        - "joint" : Perform KMeans clustering jointly on all embeddings.  
        - "independent" : Cluster each section separately.  
        - "defined" : Use externally provided cluster labels (`defined_labels`).  
    defined_labels : dict of {str: np.ndarray}, optional
        Predefined cluster labels for each section. Required if `mode="defined"`.
    vis_basis : str, default="spatial"
        Key in `adata.obsm` containing spatial coordinates for visualization.
    random_state : int, default=0
        Random seed for KMeans clustering.
    colormap : matplotlib colormap, optional
        Colormap for cluster visualization.
    swap_xy : bool, default=False
        If True, swap x and y coordinates.
    invert_x : bool, default=False
        If True, flip the image horizontally.
    invert_y : bool, default=False
        If True, flip the image vertically.
    offset : bool, default=False
        If True, shift coordinates so that the minimum is at (0, 0).
    save_path : str, optional
        Path to save the visualization images.  
        For each section, the filename will be suffixed with `_section_<ID>`.  
        If None, plots are not saved.
    dpi : int, default=300
        Resolution of the saved figures.
    remove_title : bool, default=False
        Whether to remove the figure title.
    remove_legend : bool, default=False
        Whether to hide the legend in plots.
    remove_spine : bool, default=False
        Whether to remove the axes spines.
    figscale : int, default=35
        Scaling factor for figure size.

    Returns
    -------
    dict of {str: np.ndarray}
        Cluster labels per section. Keys are section IDs, values are arrays of cluster assignments.

    Raises
    ------
    ValueError
        If `mode="defined"` but `defined_labels` is not provided.
    """

    from sklearn.cluster import KMeans
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
            remove_spine=remove_spine, 
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
    """
    Visualize clustered histology image with color-coded clusters.

    Parameters
    ----------
    he_clusters_image : ndarray of shape (H, W)
        2D array containing cluster labels for each pixel. Values should be in
        the range [0, num_he_clusters - 1].
    num_he_clusters : int
        Number of unique clusters.
    section_title : str, optional
        Title for the figure. If None, defaults to "Histology Clusters".
    colormap : list or str, optional
        - If None: use a predefined color list.  
        - If list: custom list of RGB colors (0–255) for each cluster.  
        - If str: name of a Matplotlib colormap.
    save_path : str, optional
        Path to save the figure. If None, the plot is not saved.
    figscale : int, default=35
        Scaling factor for figure size (smaller = larger figure).
    remove_title : bool, default=False
        Whether to hide the plot title.
    remove_legend : bool, default=False
        Whether to hide the cluster legend.
    remove_spine : bool, default=False
        Whether to hide the axis spines.
    dpi : int, default=300
        Resolution of the saved figure.

    Returns
    -------
    None
        Displays the histology cluster image and optionally saves it.
    """


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


import matplotlib.cm as cm
from matplotlib.colors import to_rgb
import matplotlib.patches as patches

def plot_superpixel(
    adata,
    label_key='label',  
    vis_basis='spatial',  
    colormap=None,
    save_path=None,
    save_name='visualization',
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
    Visualize superpixel clusters using labels and spatial coordinates.

    This function extracts cluster labels from `adata.obs` and spatial coordinates 
    from `adata.obsm`, reconstructs a 2D image of cluster assignments, and 
    visualizes it with color-coded clusters.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing superpixel labels in `.obs` 
        and spatial coordinates in `.obsm`.
    label_key : str, default="label"
        Key in `adata.obs` containing cluster labels.
    vis_basis : str, default="spatial"
        Key in `adata.obsm` containing spatial coordinates.
    colormap : list or str, optional
        - If None: use a predefined color palette.  
        - If list: custom list of RGB colors (0–255).  
        - If str: name of a Matplotlib colormap.
    save_path : str, optional
        Directory to save the visualization. If None, the plot is not saved.
    save_name : str, default="visualization"
        Base filename (without extension) for saving the figure.
    title : str, optional
        Title of the plot. If None, no title is shown unless `remove_title=False`.
    figscale : int, default=100
        Scaling factor for figure size (smaller = larger figure).
    format : {"png", "pdf", ...}, default="png"
        Output format for saving the figure.
    show : bool, default=True
        Whether to display the plot interactively.
    remove_title : bool, default=False
        Whether to hide the plot title.
    remove_legend : bool, default=False
        Whether to hide the cluster legend.
    remove_spine : bool, default=False
        Whether to hide the axis spines.
    dpi : int, default=300
        Resolution of the saved figure.
    random_state : int, default=2024
        Random seed for reproducibility.
    swap_xy : bool, default=False
        If True, swap x and y coordinates.
    invert_x : bool, default=False
        If True, flip the image horizontally.
    invert_y : bool, default=False
        If True, flip the image vertically.

    Returns
    -------
    None
        Displays the reconstructed cluster map and optionally saves it.
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
        [220,220,220]
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


    image_rgb = np.ones([image.shape[0], image.shape[1], 3])
    for cluster in range(num_clusters):
        image_rgb[image == cluster] = np.array(color_list[cluster]) / 255.0
    
    # image_rgb = 255 * np.ones([image.shape[0], image.shape[1], 3])
    # for cluster in range(num_clusters):
    #     image_rgb[image == cluster] = color_list[cluster]
    # image_rgb = np.array(image_rgb, dtype='uint8')
    
    plt.figure(figsize=(image.shape[1] // figscale, image.shape[0] // figscale))
    plt.imshow(image_rgb, interpolation='none')

    if remove_title or title is None:
        plt.title("")
    else:
        plt.title(title, fontsize=18)

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
        plt.savefig(file_path, dpi=dpi, bbox_inches="tight")
        print(f"Image saved to: {file_path}")


    if show:
        plt.show()
    else:
        plt.close()


# marker visualization
def prepare_image(adata, molecule_name, basis, swap_xy, invert_x, invert_y, offset, scale):
    """
    Convert molecule expression values into a 2D image based on spatial coordinates.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing molecule expression and coordinates.
    molecule_name : str
        Name of the molecule (gene or feature) to visualize.
    basis : str
        Key in `adata.obsm` containing spatial coordinates.
    swap_xy : bool
        If True, swap x and y coordinates.
    invert_x : bool
        If True, flip the image horizontally.
    invert_y : bool
        If True, flip the image vertically.
    offset : bool
        If True, shift coordinates so that the minimum is at (0, 0).
    scale : bool
        If True, scale expression values to [0, 1] using MinMaxScaler.

    Returns
    -------
    np.ndarray of shape (H, W)
        2D array where pixel intensities represent molecule expression.
        Empty positions are filled with NaN.
    """
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
    save_path: str = None,
    format: str = 'pdf'
):
    """
    Compare molecule expression between two sections as superpixel images.

    This function generates 2D expression images for the specified molecule 
    from two AnnData objects (e.g., raw vs enhanced), and displays them 
    side by side with consistent visualization settings.

    Parameters
    ----------
    molecule_name : str
        Molecule (gene/feature) name to visualize.
    adata1 : AnnData
        First annotated dataset.
    adata2 : AnnData
        Second annotated dataset.
    section1_label : str, default="Section 1"
        Title label for the first dataset.
    section2_label : str, default="Section 2"
        Title label for the second dataset.
    basis : str, default="spatial"
        Key in `.obsm` containing spatial coordinates.
    colormap : str, default="viridis"
        Colormap used for expression visualization.
    plot_style : {"original", "equal"}, default="original"
        - "original": keep default aspect ratio.  
        - "equal": enforce equal aspect ratio (square pixels).
    scale : bool, default=True
        Whether to scale expression values to [0, 1].
    swap_xy : bool, default=False
        If True, swap x and y coordinates.
    invert_x : bool, default=False
        If True, flip the image horizontally.
    invert_y : bool, default=False
        If True, flip the image vertically.
    offset : bool, default=False
        If True, shift coordinates to start at (0, 0).
    figscale : int, default=35
        Scaling factor for figure size.
    dpi : int, default=300
        Resolution of the saved figure.
    remove_title : bool, default=False
        Whether to remove subplot titles.
    remove_spine : bool, default=False
        Whether to hide plot spines.
    remove_legend : bool, default=False
        Whether to hide colorbar legends.
    save_path : str, optional
        Directory to save the figure. If None, the plot is not saved.
    format : str, default="pdf"
        Output format for saving the figure.

    Returns
    -------
    None
        Displays side-by-side expression images and optionally saves them.
    """


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
        os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(save_path, f"{molecule_name+'_combined'}.{format}")
        plt.savefig(file_path, dpi=dpi, bbox_inches="tight")
        print(f"Saving marker comparison to: {save_path}")

    plt.show()
    plt.close()


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
    
    """
    Extract log fold change (logFC) values for selected genes and groups, and return as a long-format DataFrame.

    Parameters
    ----------
    adata_list : list of AnnData
        List of annotated data matrices. Each AnnData must contain 
        differential expression results in 
        `adata.uns['rank_genes_groups']` with keys "names" and "logfoldchanges".
    adata_names : list of str
        List of dataset names corresponding to `adata_list`. Must have same length.
    target_genes : list of str
        List of genes of interest. Must have same length as `target_groups`.
    target_groups : list of str
        List of groups (clusters) corresponding to `target_genes`.
    save_path : str or Path, default="results"
        Directory where the output CSV file will be saved.
    save_name : str, default="logfc_comparison"
        Base filename (without extension) for saving.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns:
        - "Gene": target gene  
        - "Group": target group  
        - "type": dataset name  
        - "logFC": extracted log fold change (float or None if missing)

    Notes
    -----
    - The function calls `get_logfc` internally for each (gene, group, dataset).  
    - If a gene or group cannot be found in a dataset, `logFC` is recorded as None 
      and a warning is printed.  
    - The result is also saved to `{save_path}/{save_name}.csv` (tab-delimited).
    """
    
    if len(target_genes) != len(target_groups):
        raise ValueError("Length of target_genes and target_groups must match!")
    
    if len(adata_list) != len(adata_names):
        raise ValueError("Length of adata_list and adata_names must match!")
    
    results = {
        'Gene': target_genes,
        'Group': target_groups
    }
    
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
    
    logfc_df = pd.DataFrame(results)
    
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



def get_top_degs_df(
    adata: AnnData,
    n_top_genes: int = 20,
    groupby: Optional[str] = None,
    key: str = 'rank_genes_groups'
) -> pd.DataFrame:
    """
    Extract the top-N differentially expressed genes (DEGs) and their statistics 
    from Scanpy's `rank_genes_groups` results.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing differential expression results.
    n_top_genes : int, default=20
        Number of top DEGs to extract per group.
    groupby : str, optional
        Column name in `adata.obs` used for grouping. If None, it will try to read 
        from `adata.uns[key]['params']['groupby']`.
    key : str, default='rank_genes_groups'
        Key in `adata.uns` where DE results are stored.

    Returns
    -------
    pd.DataFrame
        DataFrame containing DEGs with the following columns:
        - "Group": group/cluster name
        - "Gene": gene symbol
        - "LogFC": log fold change
        - "PValue": raw p-value
        - "AdjPValue": adjusted p-value (FDR)

    Raises
    ------
    KeyError
        If `key` is not found in `adata.uns`.
    ValueError
        If required fields (`names`, `logfoldchanges`, `pvals`, `pvals_adj`) 
        are missing in `adata.uns[key]`.
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
    """
    Visualize modality weights across clusters using violin plots.

    This function plots the attention weights (stored in ``adata.obsm['alpha']``)
    for two modalities across clusters, showing their distributions as violin plots.
    Each modality is plotted with a distinct color, and the legend is placed outside
    the right side of the plot.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing modality weights in ``adata.obsm['alpha']``.
    modality_names : tuple of str, default=("RNA", "Protein")
        Names of the modalities. Must correspond to the two columns in
        ``adata.obsm['alpha']``.
    cluster_column : str, default="DePass"
        Column in ``adata.obs`` specifying cluster assignments.
    save_path : str, optional
        Directory where the plot will be saved. If None, the figure is not saved.
    save_name : str, default="modality_weights"
        Filename (without extension) for saving the plot.
    show : bool, default=True
        Whether to display the plot after creation.
    figsize : tuple of float, default=(5, 3)
        Figure size in inches.
    palette : dict, optional
        Dictionary mapping modality names to colors. If None, a default palette is used.
    **kwargs : dict
        Additional keyword arguments passed to ``seaborn.violinplot``.

    Returns
    -------
    matplotlib.axes.Axes
        The matplotlib axes object containing the plot.

    Raises
    ------
    KeyError
        If ``'alpha'`` is not found in ``adata.obsm`` or if ``cluster_column`` is not
        found in ``adata.obs``.
    ValueError
        If ``adata.obsm['alpha']`` does not have exactly two columns.
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
        linewidth=0.6,
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
            os.path.join(save_path, f"{save_name}.png"),
            dpi=300,
            bbox_inches="tight",
            transparent=True
        )
    if show:
        plt.show()
    plt.close()

    return ax


from scipy.stats import pearsonr

def calculate_correlation(adata1, adata2, gene_adt_mapping):
    """
    Calculate Pearson correlations between gene expression and ADT expression.

    This function takes two AnnData objects (one containing gene expression 
    and the other containing ADT expression) and computes Pearson correlation 
    coefficients for specified gene–ADT pairs. The mapping between genes and 
    ADTs is provided as a dictionary.

    Parameters
    ----------
    adata1 : AnnData
        AnnData object containing gene expression data.
    adata2 : AnnData
        AnnData object containing ADT expression data.
    gene_adt_mapping : dict
        Dictionary mapping ADT names (keys) to lists of gene names (values).

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per gene–ADT pair, containing:
        - ``ADT`` : ADT name
        - ``Gene`` : Gene name
        - ``Gene_ADT`` : Combined identifier in the form ``Gene_ADT``
        - ``Correlation`` : Pearson correlation coefficient
        - ``P_value`` : Two-tailed p-value for testing non-correlation

    Notes
    -----
    - Warnings are printed if a gene or ADT is not found in the corresponding dataset.
    - Assumes that ``adata1`` and ``adata2`` are aligned by cells (same observations).
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
    
            gene_expression = adata1[:, gene].X.flatten() 
            adt_expression = adata2[:, adt].X.flatten()    
            correlation, p_value = pearsonr(gene_expression, adt_expression)
            
            results.append({
                "ADT": adt,
                "Gene": gene,
                "Gene_ADT": f"{gene}_{adt}",  
                "Correlation": correlation,
                "P_value": p_value
            })
    
    return pd.DataFrame(results)