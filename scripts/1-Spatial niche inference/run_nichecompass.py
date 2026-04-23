
# The source code is borrowed from [GitHub:https://github.com/Lotfollahi-lab/nichecompass; Documentation: https://nichecompass.readthedocs.io/en/latest/] 

import os
import random
import warnings
from datetime import datetime
import gdown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import squidpy as sq
from matplotlib import gridspec
from sklearn.preprocessing import MinMaxScaler
from nichecompass.models import NicheCompass
from nichecompass.utils import (add_gps_from_gp_dict_to_adata,
                                add_multimodal_mask_to_adata,
                                create_new_color_dict,
                                extract_gp_dict_from_collectri_tf_network,
                                extract_gp_dict_from_mebocost_ms_interactions,
                                extract_gp_dict_from_nichenet_lrt_interactions,
                                extract_gp_dict_from_omnipath_lr_interactions,
                                filter_and_combine_gp_dict_gps_v2,
                                get_gene_annotations,
                                generate_multimodal_mapping_dict,
                                get_unique_genes_from_gp_dict)

dataset = "depass"
species = "mouse"
spatial_key = "spatial"
n_neighbors = 4
n_sampled_neighbors = 4
filter_genes = True
n_svg = 3000
n_svp = 15000
filter_peaks = True
min_cell_peak_thresh_ratio = 0.005
min_cell_gene_thresh_ratio = 0.005

counts_key = None
adj_key = "spatial_connectivities"
gp_names_key = "nichecompass_gp_names"
active_gp_names_key = "nichecompass_active_gp_names"
gp_targets_mask_key = "nichecompass_gp_targets"
gp_targets_categories_mask_key = "nichecompass_gp_targets_categories"
gp_sources_mask_key = "nichecompass_gp_sources"
gp_sources_categories_mask_key = "nichecompass_gp_sources_categories"
latent_key = "nichecompass_latent"

active_gp_thresh_ratio = 0.01
conv_layer_encoder = "gatv2conv"

n_epochs = 400
n_epochs_all_gps = 25
lr = 0.001
lambda_edge_recon = 500000.
lambda_gene_expr_recon = 300.
lambda_chrom_access_recon = 300.
lambda_l1_masked = 0.
lambda_l1_addon = 30.
edge_batch_size = 256
use_cuda_if_available = True

cell_type_key = "cell_type"
latent_leiden_resolution = 0.6
latent_cluster_key = f"latent_leiden_{str(latent_leiden_resolution)}"
sample_key = "batch"
spot_size = 30
differential_gp_test_results_key = "nichecompass_differential_gp_test_results"

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
now = datetime.now()
current_timestamp = now.strftime("%d%m%Y_%H%M%S")

ga_data_folder_path = "./data/gene_annotations"
gp_data_folder_path = "./data/gene_programs"
so_data_folder_path = "./depass_for_niche4k_mask01"
omnipath_lr_network_file_path = f"{gp_data_folder_path}/omnipath_lr_network.csv"
nichenet_lr_network_file_path = f"{gp_data_folder_path}/nichenet_lr_network_v2_{species}.csv"
nichenet_ligand_target_matrix_file_path = f"{gp_data_folder_path}/nichenet_ligand_target_matrix_v2_{species}.csv"
mebocost_enzyme_sensor_interactions_folder_path = f"{gp_data_folder_path}/metabolite_enzyme_sensor_gps"
collectri_tf_network_file_path = f"{gp_data_folder_path}/collectri_tf_network_{species}.csv"
marker_gp_folder_path = f"{gp_data_folder_path}/marker_gps"
gene_orthologs_mapping_file_path = f"{ga_data_folder_path}/human_mouse_gene_orthologs.csv"
gtf_file_path = f"{ga_data_folder_path}/gencode.vM25.chr_patch_hapl_scaff.annotation.gtf.gz"
artifacts_folder_path = f"./re/artifacts4k_mask01"
model_folder_path = f"{artifacts_folder_path}/{dataset+current_timestamp}/model"
figure_folder_path = f"{artifacts_folder_path}/{dataset+current_timestamp}/figures"

os.makedirs(model_folder_path, exist_ok=True)
os.makedirs(figure_folder_path, exist_ok=True)
os.makedirs(so_data_folder_path, exist_ok=True)

omnipath_gp_dict = extract_gp_dict_from_omnipath_lr_interactions(
    species=species,
    load_from_disk=False,
    save_to_disk=True,
    lr_network_file_path=omnipath_lr_network_file_path,
    gene_orthologs_mapping_file_path=gene_orthologs_mapping_file_path,
    plot_gp_gene_count_distributions=True,
    gp_gene_count_distributions_save_path=f"{figure_folder_path}/omnipath_gp_gene_count_distributions.svg")

omnipath_gp_names = list(omnipath_gp_dict.keys())
random.shuffle(omnipath_gp_names)
omnipath_gp_name = omnipath_gp_names[0]
print(f"{omnipath_gp_name}: {omnipath_gp_dict[omnipath_gp_name]}")

nichenet_gp_dict = extract_gp_dict_from_nichenet_lrt_interactions(
    species=species,
    version="v2",
    keep_target_genes_ratio=1.,
    max_n_target_genes_per_gp=250,
    load_from_disk=False,
    save_to_disk=True,
    lr_network_file_path=nichenet_lr_network_file_path,
    ligand_target_matrix_file_path=nichenet_ligand_target_matrix_file_path,
    gene_orthologs_mapping_file_path=gene_orthologs_mapping_file_path,
    plot_gp_gene_count_distributions=True)

nichenet_gp_names = list(nichenet_gp_dict.keys())
random.shuffle(nichenet_gp_names)
nichenet_gp_name = nichenet_gp_names[0]
print(f"{nichenet_gp_name}: {nichenet_gp_dict[nichenet_gp_name]}")

mebocost_gp_dict = extract_gp_dict_from_mebocost_ms_interactions(
    dir_path=mebocost_enzyme_sensor_interactions_folder_path,
    species=species,
    plot_gp_gene_count_distributions=True)

mebocost_gp_names = list(mebocost_gp_dict.keys())
random.shuffle(mebocost_gp_names)
mebocost_gp_name = mebocost_gp_names[0]
print(f"{mebocost_gp_name}: {mebocost_gp_dict[mebocost_gp_name]}")

collectri_gp_dict = extract_gp_dict_from_collectri_tf_network(
        species=species,
        tf_network_file_path=collectri_tf_network_file_path,
        load_from_disk=False,
        save_to_disk=True,
        plot_gp_gene_count_distributions=True)

collectri_gp_names = list(collectri_gp_dict.keys())
random.shuffle(collectri_gp_names)
collectri_gp_name = collectri_gp_names[0]
print(f"{collectri_gp_name}: {collectri_gp_dict[collectri_gp_name]}")

gp_dicts = [omnipath_gp_dict, nichenet_gp_dict, mebocost_gp_dict, collectri_gp_dict]
combined_gp_dict = filter_and_combine_gp_dict_gps_v2(
    gp_dicts,
    verbose=True)

print(f"Number of gene programs after filtering and combining: {len(combined_gp_dict)}.")

adata = sc.read_h5ad(f"{so_data_folder_path}/{dataset}.h5ad")
adata_atac = sc.read_h5ad(f"{so_data_folder_path}/{dataset}_atac.h5ad")

cell_type_df = pd.read_csv(f"{so_data_folder_path}/{dataset}_cell_type_annotations.csv", index_col=0)
cell_type_df.rename({"predicted.celltype": cell_type_key}, axis=1, inplace=True)
cell_type_df.drop("ATAC_clusters", axis=1, inplace=True)
adata.obs = adata.obs.merge(cell_type_df, left_index=True, right_index=True, how="left")

sq.gr.spatial_neighbors(adata,
                        coord_type="generic",
                        spatial_key=spatial_key,
                        n_neighs=n_neighbors)

adata.obsp[adj_key] = (
    adata.obsp[adj_key].maximum(
        adata.obsp[adj_key].T))

if filter_genes:
    print("Filtering genes...")
    gp_dict_genes = get_unique_genes_from_gp_dict(
        gp_dict=combined_gp_dict,
            retrieved_gene_entities=["sources", "targets"])
    print(f"Starting with {len(adata.var_names)} genes.")
    min_cells = int(adata.shape[0] * min_cell_gene_thresh_ratio)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    print(f"Keeping {len(adata.var_names)} genes after filtering genes with counts in less than {int(adata.shape[0] * min_cell_gene_thresh_ratio)} cells.")
    
    sq.gr.spatial_autocorr(adata, mode="moran", genes=adata.var_names)
    svg_genes = adata.uns["moranI"].index[:n_svg].tolist()
    adata.var["spatially_variable"] = adata.var_names.isin(svg_genes)
    adata = adata[:, adata.var["spatially_variable"] == True]
    print(f"Keeping {len(adata.var_names)} spatially variable genes.")
    
if filter_peaks:
    print("\nFiltering peaks...")
    print(f"Starting with {len(adata_atac.var_names)} peaks.")
    min_cells = int(adata_atac.shape[0] * min_cell_peak_thresh_ratio)
    sc.pp.filter_genes(adata_atac, min_cells=min_cells)
    print(f"Keeping {len(adata_atac.var_names)} peaks after filtering peaks with counts in less than {int(adata_atac.shape[0] * min_cell_peak_thresh_ratio)} cells.")
    
    adata_atac.obsp["spatial_connectivities"] = adata.obsp["spatial_connectivities"]
    adata_atac.obsp["spatial_distances"] = adata.obsp["spatial_distances"]

    sq.gr.spatial_autocorr(adata_atac,
                           mode="moran",
                           genes=adata_atac.var_names)
    sv_peaks = adata_atac.uns["moranI"].index[:n_svp].tolist()
    adata_atac.var["spatially_variable"] = adata_atac.var_names.isin(sv_peaks)
    adata_atac = adata_atac[:, adata_atac.var["spatially_variable"] == True]
    print(f"Keeping {len(adata_atac.var_names)} peaks after filtering spatially variable peaks.")

adata, adata_atac = get_gene_annotations(
    adata=adata,
    adata_atac=adata_atac,
    gtf_file_path=gtf_file_path)

add_gps_from_gp_dict_to_adata(
    gp_dict=combined_gp_dict,
    adata=adata,
    gp_targets_mask_key=gp_targets_mask_key,
    gp_targets_categories_mask_key=gp_targets_categories_mask_key,
    gp_sources_mask_key=gp_sources_mask_key,
    gp_sources_categories_mask_key=gp_sources_categories_mask_key,
    gp_names_key=gp_names_key,
    min_genes_per_gp=2,
    min_source_genes_per_gp=0,
    min_target_genes_per_gp=1,
    max_genes_per_gp=None,
    max_source_genes_per_gp=None,
    max_target_genes_per_gp=None,
    plot_gp_gene_count_distributions=True)

gene_peak_mapping_dict = generate_multimodal_mapping_dict(
    adata=adata,
    adata_atac=adata_atac)
adata, adata_atac = add_multimodal_mask_to_adata(
    adata=adata,
    adata_atac=adata_atac,
    gene_peak_mapping_dict=gene_peak_mapping_dict)

print(f"Keeping {len(adata_atac.var_names)} peaks after filtering peaks with no matching genes in gp mask.")

model = NicheCompass(adata,
                     adata_atac,
                     counts_key=counts_key,
                     adj_key=adj_key,
                     gp_names_key=gp_names_key,
                     active_gp_names_key=active_gp_names_key,
                     gp_targets_mask_key=gp_targets_mask_key,
                     gp_targets_categories_mask_key=gp_targets_categories_mask_key,
                     gp_sources_mask_key=gp_sources_mask_key,
                     gp_sources_categories_mask_key=gp_sources_categories_mask_key,
                     latent_key=latent_key,
                     conv_layer_encoder=conv_layer_encoder,
                     active_gp_thresh_ratio=active_gp_thresh_ratio)

model.train(n_epochs=n_epochs, 
            n_epochs_all_gps=n_epochs_all_gps,
            lr=lr,
            lambda_edge_recon=lambda_edge_recon,
            lambda_gene_expr_recon=lambda_gene_expr_recon,
            lambda_chrom_access_recon=lambda_chrom_access_recon,
            lambda_l1_masked=lambda_l1_masked,
            lambda_l1_addon=lambda_l1_addon,
            edge_batch_size=edge_batch_size,
            use_cuda_if_available=use_cuda_if_available,
            n_sampled_neighbors=n_sampled_neighbors,
            verbose=True)

sc.pp.neighbors(model.adata,
                use_rep=latent_key,
                key_added=latent_key)

sc.tl.umap(model.adata,
           neighbors_key=latent_key)

model.save(dir_path=model_folder_path,
           overwrite=True,
           save_adata=True,
           adata_file_name="adata.h5ad",
           save_adata_atac=True,
           adata_atac_file_name=f"adata_atac.h5ad")

load_timestamp = dataset+current_timestamp

figure_folder_path = f"{artifacts_folder_path}/{load_timestamp}/figures"
model_folder_path = f"{artifacts_folder_path}/{load_timestamp}/model"

model.adata.uns = {}
model.adata.write(model_folder_path+'/nichecompass.h5ad')

from nichecompass.models import NicheCompass

model = NicheCompass.load(dir_path=model_folder_path,
                          adata=None,
                          adata_file_name="adata.h5ad",
                          adata_atac=None,
                          adata_atac_file_name="adata_atac.h5ad",
                          gp_names_key=gp_names_key)
samples = model.adata.obs[sample_key].unique().tolist()

cell_type_colors = create_new_color_dict(
    adata=model.adata,
    skip_default_colors=50,
    cat_key=cell_type_key)

groups = None
save_fig = True
file_path = f"{figure_folder_path}/cell_types_latent_physical_space.svg"

fig = plt.figure(figsize=(12, 14))
title = fig.suptitle(t="Cell Types in Latent and Physical Space", y=0.96, x=0.55, fontsize=20)
spec1 = gridspec.GridSpec(ncols=1, nrows=2, width_ratios=[1], height_ratios=[3, 2])
spec2 = gridspec.GridSpec(ncols=len(samples), nrows=2, width_ratios=[1] * len(samples), height_ratios=[3, 2])
axs = []
axs.append(fig.add_subplot(spec1[0]))
sc.pl.umap(adata=model.adata,
           color=[cell_type_key],
           groups=groups,palette=cell_type_colors,
           title=f"Cell Types in Latent Space",
           ax=axs[0],
           show=False)
for idx, sample in enumerate(samples):
    axs.append(fig.add_subplot(spec2[len(samples) + idx]))
    sc.pl.spatial(adata=model.adata[model.adata.obs[sample_key] == sample],
                  color=[cell_type_key],
                  groups=groups,
                  palette=cell_type_colors,
                  spot_size=spot_size,
                  title=f"Cell Types in Physical Space \n(Sample: {sample})",
                  legend_loc=None,
                  ax=axs[idx+1],
                  show=False)

handles, labels = axs[0].get_legend_handles_labels()
lgd = fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.98, 0.5))
axs[0].get_legend().remove()

plt.subplots_adjust(wspace=0.2, hspace=0.25)
if save_fig:
    fig.savefig(file_path, bbox_extra_artists=(lgd, title), bbox_inches="tight")
plt.show()

latent_leiden_resolution = 1
sc.tl.leiden(adata=model.adata,
             resolution=latent_leiden_resolution,
             key_added=latent_cluster_key,
             neighbors_key=latent_key)
latent_cluster_colors = create_new_color_dict(
    adata=model.adata,
    cat_key=latent_cluster_key)

groups = None
save_fig = True
file_path = f"{figure_folder_path}/res_{latent_leiden_resolution}_niches_latent_physical_space.svg"

fig = plt.figure(figsize=(12, 14))
title = fig.suptitle(t=f"NicheCompass Niches in Latent and Physical Space", y=0.96, x=0.55, fontsize=20)
spec1 = gridspec.GridSpec(ncols=1, nrows=2, width_ratios=[1], height_ratios=[3, 2])
spec2 = gridspec.GridSpec(ncols=len(samples), nrows=2, width_ratios=[1] * len(samples), height_ratios=[3, 2])
axs = []
axs.append(fig.add_subplot(spec1[0]))
sc.pl.umap(adata=model.adata,
           color=[latent_cluster_key],
           groups=groups,
           palette=latent_cluster_colors,
           title=f"Niches in Latent Space",
           ax=axs[0],
           show=False)
for idx, sample in enumerate(samples):
    axs.append(fig.add_subplot(spec2[len(samples) + idx]))
    sc.pl.spatial(adata=model.adata[model.adata.obs[sample_key] == sample],
                  color=[latent_cluster_key],
                  groups=groups,
                  palette=latent_cluster_colors,
                  spot_size=spot_size,
                  title=f"Niches in Physical Space \n(Sample: {sample})",
                  legend_loc=None,
                  ax=axs[idx+1],
                  show=False)

handles, labels = axs[0].get_legend_handles_labels()
lgd = fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.98, 0.5))
axs[0].get_legend().remove()

plt.subplots_adjust(wspace=0.2, hspace=0.25)
if save_fig:
    fig.savefig(file_path, bbox_extra_artists=(lgd, title), bbox_inches="tight")
