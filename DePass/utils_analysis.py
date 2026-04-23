
import os
import warnings
from pathlib import Path
from typing import Optional, Union, Tuple, List, Dict, Sequence
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from matplotlib import cm, patches, font_manager
from matplotlib.colors import to_rgb
import seaborn as sns
import scanpy as sc
import anndata as ad
from anndata import AnnData
from scipy.stats import pearsonr

mpl.rcParams['pdf.fonttype'] = 42


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
    Plot spatial data using ``scanpy.pl.embedding``.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix containing spatial coordinates in
        ``adata.obsm['spatial']``.

    color : str, optional
        Column name in ``adata.obs`` or gene name used to color the plot.
        Default is ``"DePass"``.

    save_path : str or pathlib.Path, optional
        Directory where the figure will be saved.
        If ``None``, the figure is not saved.

    save_name : str, optional
        Filename (without extension) for saving the plot.
        Default is ``"spatial_plot"``.

    title : str, optional
        Title of the plot. If ``None``, defaults to the ``color`` argument.

    s : int, optional
        Marker size. Default is ``35``.

    figsize : tuple of float, optional
        Figure size in inches. Default is ``(3, 3)``.

    dpi : int, optional
        Resolution of the saved figure. Default is ``300``.

    format : {"png", "pdf", "svg", "tiff", "jpg", "jpeg"}, optional
        Output file format. Default is ``"png"``.

    frameon : bool, optional
        Whether to show a frame around the plot. Default is ``True``.

    adjust_margins : bool, optional
        Whether to tighten layout by adjusting margins. Default is ``True``.

    legend_loc : str or None, optional
        Position of the legend. Set to ``None`` to disable.
        Default is ``"right margin"``.

    colorbar_loc : str or None, optional
        Position of the colorbar. Set to ``None`` to disable.
        Default is ``"right"``.

    show : bool, optional
        Whether to display the plot interactively. Default is ``False``.

    **kwargs
        Additional keyword arguments passed to ``scanpy.pl.embedding``.


    Returns
    -------
    None
        Generates a spatial plot and optionally saves it to disk.


    Notes
    -----
    This function can be used to visualize spatial patterns
    based on clustering results obtained from
    ``DePass.utils.clustering``.

    Cluster labels are typically stored in ``adata.obs``
    after clustering.

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
    Plot expression of selected marker genes across groups using spatial embedding plots.

    Parameters
    ----------
    adata1 : AnnData
        Raw annotated data matrix.
    adata2 : AnnData
        Enhanced annotated data matrix.
    target_gene : str
        Target gene to visualize.
    save_path : str, optional
        Directory where the figure will be saved. If None, the figure is not saved.
    save_name : str, default="gene_comparison"
        Filename for saving the figure.
    show : bool, default=False
        Whether to display the plot.
    s : int, default=80
        Size of the points in the spatial plot.
    cmap : str, default="turbo"
        Colormap for the plot.
    dpi : int, default=300
        Resolution of the saved figure.
    colorbar_loc : str, optional
        Location of the color bar.
    figsize : tuple of float, default=(7, 3)
        Figure size in inches.
    frameon : bool, default=False
        Whether to show the frame around the plot.

    Returns
    -------
    None
        The function generates spatial plots and optionally saves them.

    Notes
    -----
    This function visualizes and compares gene expression between raw and enhanced
    spatial omics data. The enhanced data is generated by the DePass model after
    running ``DePass.train()``.

    Spatial coordinates must be stored in ``adata.obsm['spatial']``.
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
        'frameon': frameon,
        'colorbar_loc': colorbar_loc,
        'cmap': cmap,
    }

    def _scaler_data(adata: sc.AnnData) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            expr = adata[:, target_gene].X.toarray()
        adata.obs[f'{target_gene}_expr'] = MinMaxScaler().fit_transform(expr)

    def _create_plot(adata: sc.AnnData, ax: plt.Axes, name: str) -> None:
        sc.pl.embedding(
            adata,
            title=f"{name}{target_gene}",
            ax=ax,
            show=False,
            **vis_params
        )

    _scaler_data(adata1)
    _scaler_data(adata2)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    _create_plot(adata1, axes[0], 'Raw - ')
    _create_plot(adata2, axes[1], 'Enhanced - ')

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
        The function generates spatial plots of the target gene in both datasets,
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
        fig.savefig(
            os.path.join(save_path, f"{save_name}_combined_logFC.png"),
            dpi=dpi,
            bbox_inches="tight"
        )
    if show:
        plt.show()
    plt.close(fig)





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
    invert_y=False,
    default_color_dict=None
):
    """
    Visualize superpixel clusters using labels and spatial coordinates.

    This function extracts cluster labels from `adata.obs` and spatial coordinates 
    from `adata.obsm`, reconstructs a 2D image of cluster assignments, and 
    visualizes it with color-coded clusters. Supports custom color mapping, 
    coordinate transformation, and high-quality saving.

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
        Unused in this version; retained for compatibility.
    save_path : str, optional
        Directory to save the visualization. If None, the plot is not saved.
    save_name : str, default="visualization"
        Base filename (without extension) for saving the figure.
    title : str, optional
        Title of the plot.
    figscale : int, default=100
        Scaling factor for figure size (smaller = larger figure).
    format : str, default="png"
        Output format for saving the figure (e.g., png, pdf).
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
    default_color_dict : dict, optional
        Custom color dictionary mapping label values to RGB lists (0-255).
        If None, a built-in professional color palette is used.

        


    Returns
    -------
    None
        Displays the reconstructed cluster map and optionally saves it.


    Notes
    -----
    This function visualizes superpixel-level spatial patterns
    based on clustering results obtained from
    ``DePass.utils.clustering``.

    Cluster labels are typically stored in ``adata.obs``
    after clustering.


    """

    np.random.seed(random_state)


    if default_color_dict is None:
        default_color_dict = {
            0: [220, 220, 220], 1: [44, 160, 44], 2: [255, 187, 120], 3: [188, 189, 34], 4: [140, 86, 75],
            5: [22, 255, 255], 6: [127, 127, 127], 7: [180, 100, 225], 8: [23, 190, 207], 9: [174, 199, 232],
            10: [255, 100, 255], 11: [60, 255, 90], 12: [241, 91, 108], 13: [255, 204, 0], 14: [196, 156, 148],
            15: [210, 189, 142], 16: [199, 199, 199], 17: [255, 188, 188], 18: [158, 218, 229], 19: [128, 100, 243],
            20: [60, 162, 254], 21: [204, 238, 204], 22: [254, 220, 189], 23: [197, 176, 213], 24: [230, 190, 255],
            25: [255, 127, 14], 26: [157, 201, 42], 27: [113, 187, 55], 28: [200, 219, 78], 29: [255, 228, 181],
            30: [0, 128, 255], -1: [240, 240, 240]
        }


    labels = adata.obs[label_key].values
    coords = adata.obsm[vis_basis].copy().astype(int)


    if swap_xy:
        coords = coords[:, [1, 0]]

    valid_labels = labels
    valid_coords = coords


    unique_labels = np.unique(valid_labels)
    color_list_final = [default_color_dict[label] for label in unique_labels]
    label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}


    max_y, max_x = coords.max(axis=0) + 1
    image = np.full((max_y, max_x), fill_value=-1, dtype=int)

    for (y, x), lbl in zip(valid_coords, valid_labels):
        if 0 <= x < max_x and 0 <= y < max_y:
            image[y, x] = label_to_idx[lbl]


    if invert_x:
        image = image[:, ::-1]
    if invert_y:
        image = image[::-1, :]


    image_rgb = np.ones((image.shape[0], image.shape[1], 3))
    for i, lbl in enumerate(unique_labels):
        image_rgb[image == i] = np.array(color_list_final[i]) / 255.0


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

    # Legend
    if not remove_legend:
        legend_elements = [
            patches.Patch(
                facecolor=np.array(color_list_final[i]) / 255,
                label=f'Cluster {unique_labels[i]}'
            ) for i in range(len(unique_labels))
        ]
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)


    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(
            os.path.join(save_path, f"{save_name}.{format}"),
            dpi=dpi,
            bbox_inches="tight"
        )

    if show:
        plt.show()
    else:
        plt.close()






def plot_superpixel_str(
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
    Visualize superpixel clusters using categorical labels
    (e.g. 'Tumor', 'Stroma', '0', '1', 'A', 'B').

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix.

    label_key : str, optional
        Key in ``adata.obs`` containing cluster labels.
        Default is ``"label"``.

    vis_basis : str, optional
        Key in ``adata.obsm`` specifying spatial coordinates.
        Default is ``"spatial"``.

    colormap : list or str, optional
        Colormap used for cluster visualization.

    save_path : str, optional
        Directory to save the plot. If ``None``, the figure is not saved.

    save_name : str, optional
        Filename (without extension). Default is ``"visualization"``.

    title : str, optional
        Title of the plot.

    figscale : int, optional
        Scaling factor controlling figure size. Default is ``100``.

    format : str, optional
        Output file format. Default is ``"png"``.

    show : bool, optional
        Whether to display the plot. Default is ``True``.

    remove_title : bool, optional
        Whether to remove the title. Default is ``False``.

    remove_legend : bool, optional
        Whether to remove the legend. Default is ``False``.

    remove_spine : bool, optional
        Whether to remove axis spines. Default is ``False``.

    dpi : int, optional
        Resolution of the saved figure. Default is ``300``.

    random_state : int, optional
        Random seed for reproducibility. Default is ``2024``.

    swap_xy : bool, optional
        Whether to swap x and y axes. Default is ``False``.

    invert_x : bool, optional
        Whether to invert the x-axis. Default is ``False``.

    invert_y : bool, optional
        Whether to invert the y-axis. Default is ``False``.


    Returns
    -------
    None
        Generates a categorical spatial visualization.
    """


    np.random.seed(random_state)
    
    
    labels = adata.obs[label_key].astype(str).values
    coords = adata.obsm[vis_basis].copy().astype(int)
    
    if swap_xy:
        coords = coords[:, [1, 0]]

    unique_labels = np.unique(labels)
    num_clusters = len(unique_labels)

    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    zero_based_labels = np.array([label_to_index[label] for label in labels])


    if colormap is None:
        color_list = [
            [220, 220, 220], [44, 160, 44], [255, 187, 120], [188, 189, 34], [140, 86, 75],
            [22, 255, 255], [127, 127, 127], [180, 100, 225], [23, 190, 207], [174, 199, 232],
            [255, 100, 255], [60, 255, 90], [241, 91, 108], [255, 204, 0], [196, 156, 148],
            [210, 189, 142], [199, 199, 199], [255, 188, 188], [158, 218, 229], [128, 100, 243],
            [60, 162, 254], [204, 238, 204], [254, 220, 189], [197, 176, 213], [230, 190, 255],
            [255, 127, 14], [157, 201, 42], [113, 187, 55], [200, 219, 78], [255, 228, 181],
            [0, 128, 255], [240, 240, 240]
        ]
    elif isinstance(colormap, list):
        color_list = colormap
    else:
        cmap = cm.get_cmap(colormap, num_clusters)
        color_list = [[int(255 * c) for c in to_rgb(cmap(i))] for i in range(num_clusters)]


    if len(color_list) < num_clusters:
        base_colors = np.array(color_list)
        extra_colors = base_colors[np.random.choice(len(base_colors), num_clusters - len(base_colors))]
        color_list = list(base_colors) + list(extra_colors)


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
    for cluster_idx in range(num_clusters):
        image_rgb[image == cluster_idx] = np.array(color_list[cluster_idx]) / 255.0


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
        legend_elements = [
            patches.Patch(
                facecolor=np.array(color_list[i]) / 255,
                label=f'{unique_labels[i]}'
            ) for i in range(num_clusters)
        ]
        plt.legend(
            handles=legend_elements,
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            borderaxespad=0.,
            fontsize=12
        )

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

    """

        
    def _prepare_image(adata, molecule_name, basis, swap_xy, invert_x, invert_y, offset, scale):
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


    img1 = _prepare_image(adata1, molecule_name, basis, swap_xy, invert_x, invert_y, offset, scale)
    img2 = _prepare_image(adata2, molecule_name, basis, swap_xy, invert_x, invert_y, offset, scale)


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
    modality_names: Sequence[str],
    cluster_column: str = "DePass",
    sort_clusters: bool = True,
    numeric_sort: bool = True,
    save_path: Optional[str] = None,
    save_name: str = "modality_weights",
    show: bool = True,
    figsize: tuple = (5, 3),
    palette: Optional[Dict[str, str]] = None,
    title: str = None,
    smoothing_method: Optional[str] = 'temperature',
    dirichlet_alpha: float = 0.1,
    temperature_T: float = 1.0,
    uniform_lambda: float = 0.2,
    ylim: bool = True,
    ylim1: float = 0.,
    ylim2: float = 0.95,
    fontsize: float = 10,
    **kwargs
) -> plt.Axes:
    """
    Visualize modality attention weight distributions across clusters
    using violin plots.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing modality attention weights in
        ``adata.obsm['alpha']``.

    modality_names : Sequence[str]
        Names of modalities corresponding to columns in ``adata.obsm['alpha']``.

    cluster_column : str
        Column in ``adata.obs`` containing cluster labels.

    sort_clusters : bool, optional
        Whether to sort clusters. Default is ``True``.

    numeric_sort : bool, optional
        Whether to sort clusters numerically (if applicable).
        Default is ``False``.

    save_path : str, optional
        Directory to save the plot. If ``None``, the figure is not saved.

    save_name : str, optional
        Base filename (without extension) for saving the figure.
        Default is ``"violin_plot"``.

    show : bool, optional
        Whether to display the plot. Default is ``True``.

    figsize : tuple of float, optional
        Figure size in inches.

    palette : dict, optional
        Color mapping for modalities.

    title : str, optional
        Title of the plot.

    smoothing_method : {"dirichlet", "temperature", "uniform"}, optional
        Method used to smooth modality weights.

    dirichlet_alpha : float, optional
        Concentration parameter for Dirichlet smoothing.

    temperature_T : float, optional
        Temperature parameter for scaling-based smoothing.

    uniform_lambda : float, optional
        Mixing weight for uniform smoothing.

    ylim : bool, optional
        Whether to apply y-axis limits. Default is ``False``.

    ylim1 : float, optional
        Lower bound of the y-axis.

    ylim2 : float, optional
        Upper bound of the y-axis.

    fontsize : float, optional
        Base font size.

    **kwargs
        Additional keyword arguments passed to ``seaborn.violinplot``.


    Returns
    -------
    None
        Generates a violin plot showing modality weight distributions
        across clusters.
    """
    mpl.rcParams.update({
        'axes.edgecolor': 'black',
        'axes.labelcolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'text.color': 'black'
    })
    mpl.rcParams['pdf.fonttype'] = 42

    font_manager.fontManager.addfont('/home/jyx2/DePass-main/fonts/ARIAL.TTF')
    font_manager.fontManager.addfont('/home/jyx2/DePass-main/fonts/ARIALBD.TTF')
    plt.rcParams['font.family'] = 'Arial'

    plt.rcParams.update({
        'font.size': fontsize,
        'axes.titlesize': fontsize,
        'axes.labelsize': fontsize,
        'xtick.labelsize': fontsize,
        'ytick.labelsize': fontsize,
        'legend.fontsize': fontsize - 4
    })

    sns.set_style("white")
    sns.set_context("notebook")

    if "alpha" not in adata.obsm:
        raise KeyError("Missing modality weights in adata.obsm['alpha']")

    if cluster_column not in adata.obs:
        raise KeyError(f"Cluster column '{cluster_column}' not found in adata.obs")

    alpha = adata.obsm["alpha"].copy()

    if len(modality_names) != alpha.shape[1]:
        raise ValueError("modality_names does not match alpha dimension")

    if smoothing_method is not None:
        alpha = _apply_smoothing(
            alpha,
            method=smoothing_method,
            dirichlet_alpha=dirichlet_alpha,
            temperature_T=temperature_T,
            uniform_lambda=uniform_lambda
        )

    plot_df = pd.DataFrame(alpha, columns=modality_names, index=adata.obs_names)
    plot_df["Cluster"] = adata.obs[cluster_column]

    if sort_clusters:
        try:
            clusters = sorted(plot_df["Cluster"].unique(), key=lambda x: float(x))
        except:
            clusters = sorted(plot_df["Cluster"].unique())

        plot_df["Cluster"] = pd.Categorical(
            plot_df["Cluster"], categories=clusters, ordered=True
        )

    melted_df = plot_df.melt(
        id_vars="Cluster",
        value_vars=modality_names,
        var_name="Modality",
        value_name="Weight"
    )

    if palette is None:
        palette = {
            "RNA": "#F48888",
            "Protein": "#7AA8E0",
            "ATAC": "#8BE0A8",
            "Matabolate": "#B495E0",
            "H3K4me3": "#F6B080",
            "H3K27me3": "#77C8C0"
        }

    plt.figure(figsize=figsize)

    ax = sns.violinplot(
        data=melted_df,
        x="Cluster",
        y="Weight",
        hue="Modality",
        inner="quart",
        linewidth=0.8,
        palette=palette,
        **kwargs
    )

    if title is None:
        title = "Modality Attention Weights"
        if smoothing_method:
            title += f" ({smoothing_method} smoothed)"

    ax.set_title(title, pad=15, weight='bold')
    ax.set_xlabel("Cluster", labelpad=10)
    ax.set_ylabel("Modality Weight", labelpad=10)

    if ylim:
        ax.set_ylim(ylim1, ylim2)

    ax.set_xticks(range(len(plot_df["Cluster"].cat.categories)))
    ax.set_xticklabels(plot_df["Cluster"].cat.categories)

    ax.yaxis.set_major_locator(plt.MaxNLocator(5))

    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    ax.tick_params(
        axis='both',
        which='both',
        length=6,
        width=1,
        direction='out',
        colors='black'
    )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.spines['left'].set_color("black")
    ax.spines['bottom'].set_color("black")

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        title="Modality"
    )

    plt.subplots_adjust(right=0.8)
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




def _apply_smoothing(alpha: np.ndarray, method: str, **kwargs) -> np.ndarray:
    for i in range(alpha.shape[0]):
        p = alpha[i, :].copy()
        
        if method == 'dirichlet':
            alpha_val = kwargs.get('dirichlet_alpha', 0.1)
            p_smooth = p + alpha_val
            p_smooth /= p_smooth.sum()
            
        elif method == 'temperature':
            T = kwargs.get('temperature_T', 2.0)
            p_safe = np.clip(p, 1e-10, None)
            p_smooth = p_safe ** (1 / T)
            p_smooth /= p_smooth.sum()
            
        elif method == 'uniform':
            lam = kwargs.get('uniform_lambda', 0.2)
            K = len(p)
            uniform = np.ones(K) / K
            p_smooth = (1 - lam) * p + lam * uniform
            
        else:
            p_smooth = p
            
        alpha[i, :] = p_smooth
        
    return alpha





def calculate_correlation(adata1, adata2, gene_adt_mapping):
    """
    Calculate Pearson correlations between gene expression and ADT expression.

    This function takes two AnnData objects (one containing gene expression 
    and the other containing ADT expression) and computes Pearson correlation 
    coefficients for specified gene-ADT pairs. The mapping between genes and 
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
        DataFrame with one row per gene-ADT pair, containing:
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





def truncate_expression_smartclip(
    adata,
    name,
    lower=0,
    upper=99,
    source='var'
):
    """
    Truncate expression values using percentile-based clipping to reduce outliers.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix.

    name : str
        Feature name. Interpreted as:

        - gene name if ``source='var'``
        - observation key if ``source='obs'``

    lower : float, optional
        Lower percentile for truncation (range: 0–100).
        Only applied to non-zero values. Default is ``0``.

    upper : float, optional
        Upper percentile for truncation (range: 0–100).
        Only applied to non-zero values. Default is ``99``.

    source : {"var", "obs"}, optional
        Data source for extracting values:

        - ``"var"`` : gene expression from ``adata.X``
        - ``"obs"`` : values from ``adata.obs``

        Default is ``"var"``.

    Returns
    -------
    anndata.AnnData
        A new AnnData object containing the truncated values:

        - ``X`` : clipped values reshaped as (n_cells, 1)
        - ``obs`` : copied from input ``adata``
        - ``var`` : corresponding feature metadata

    Notes
    -----
    Clipping is performed based on percentiles computed from
    non-zero values only, which avoids distortion from zero inflation
    commonly observed in single-cell and spatial omics data.

    Values below and above the specified percentiles are clipped
    to the corresponding thresholds.
    """
    if source == 'var':
        if name not in adata.var_names:
            raise ValueError(f"{name} not in adata.var_names")
        values = adata[:, name].X

    elif source == 'obs':
        if name not in adata.obs:
            raise ValueError(f"{name} not in adata.obs")
        values = adata.obs[name].values

    else:
        raise ValueError("source must be 'var' or 'obs'")

    if hasattr(values, "toarray"):
        values = values.toarray().flatten()
    else:
        values = np.array(values).flatten()

    nonzero_values = values[values > 0]
    num_nonzero = len(nonzero_values)

    if num_nonzero == 0:
        values_clipped = values
    else:
        vmin = np.percentile(nonzero_values, lower) if lower > 0 else 0
        vmax = np.percentile(nonzero_values, upper)
        values_clipped = np.clip(values, vmin, vmax)

    if source == 'var':
        new_adata = ad.AnnData(
            X=values_clipped[:, np.newaxis],
            obs=adata.obs.copy(),
            var=adata[:, name].var.copy(),
            obsm=adata.obsm.copy(),
            uns=adata.uns.copy()
        )

    else:
        new_adata = ad.AnnData(
            X=values_clipped[:, np.newaxis],
            obs=adata.obs.copy(),
            var=pd.DataFrame(index=[name]),
            obsm=adata.obsm.copy(),
            uns=adata.uns.copy()
        )
        new_adata.obs[name] = values_clipped

    return new_adata