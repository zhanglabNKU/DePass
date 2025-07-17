import torch
from tqdm import tqdm
import torch.nn.functional as F
import pandas as pd
import numpy as np
import time
from .Net import DePassAE, DePassAE_va
from sklearn.neighbors import kneighbors_graph
from .utils import *
from torch_geometric.loader import ClusterData, ClusterLoader
from sklearn.decomposition import PCA
from scipy.sparse import coo_matrix
from torch.optim.lr_scheduler import MultiStepLR

class DePass:
    def __init__(
                 self,
                 data: dict,
                 data_type: str = 'spatial',
                 device: torch.device = torch.device('cpu'),
                 learning_rate: float = 2e-4,
                 epochs: int = 200,
                 batch_training: bool = False,
                 sub_graph_size: int = 1024,
                 updata_step: int = 25,
                 mlpLayer1: list = [256, 64],
                 mlpLayer2: list = [256, 64],
                 mlpLayer3: list = [256, 64],
                 dim: int = 64,
                 K_spatial: int = 5,
                 K_feature: int = 20,
                 ):
        
        r"""
        Initialize the DePass model.

        Args:
            data: dict: Input data dictionary containing multiple modality data.
            data_type: str='spatial': Data type, which can be 'spatial' or 'single_cell'.
            device: torch.device=torch.device('cpu'): Computing device.
            learning_rate: float=2e-4: Learning rate.
            epochs: int=200: Total number of training epochs.
            batch_training: bool=False: Whether to use batch training.
            sub_graph_size: int=1024: sub graph size for batch training.
            updata_step: int=25: Update step.
            mlpLayer1: list=[256, 64]: Structure of the first MLP layer.
            mlpLayer2: list=[256, 64]: Structure of the second MLP layer.
            mlpLayer3: list=[256, 64]: Structure of the third MLP layer.
            dim: int=64: Embedding dimension.
            K_spatial: int=5: Number of neighbors for the spatial KNN graph.
            K_feature: int=20: Number of neighbors for the feature KNN graph.
        """
        self.data = data.copy()
        self.data_type = data_type
        self.device = device
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_training = batch_training
        self.sub_graph_size = sub_graph_size
        self.updata_step = updata_step
        self.mlpLayer1 = mlpLayer1
        self.mlpLayer2 = mlpLayer2
        self.mlpLayer3 = mlpLayer3
        self.dim = dim
        self.K_spatial = K_spatial
        self.K_feature = K_feature

        self.n_views = len(self.data)

        # if self.n_views not in [2, 3]:
        #     raise ValueError("The number of input modalities is more than 3. We suggest integrating 2 or 3 at a time, but got {}".format(self.n_views))
       
        if self.data_type not in ['spatial', 'single_cell']:
            raise ValueError("The data type must be 'spatial' or 'single_cell', but got {}".format(self.data_type))
        
        first_key = next(iter(self.data.keys()))
        if self.data_type == 'spatial':
            if 'spatial' not in self.data[first_key].obsm:
                raise ValueError("The 'spatial' key is not found in `adata.obsm`. Please provide the spatial coordinates.")
            
        if self.device.type == 'cuda':
           device_name = torch.cuda.get_device_name(0)
        else:
           device_name ='CPU'

        print('[Config]')
        print('Modalities:',self.n_views,'| Data:',self.data_type,'| Device:',device_name,'\n')   

    def train(self):
        if self.n_views == 2:
            modality = list(self.data.keys())
            self.mlpLayer1=[128,64] if modality[0] in ['protein'] else [256,64]
            self.mlpLayer2=[128,64] if modality[1] in ['protein'] else [256,64]
            self.adata_omics1 = self.data[modality[0]]
            self.adata_omics2 = self.data[modality[1]]

            self.features_omics1 = torch.FloatTensor(self.adata_omics1.obsm['input_feat'].copy()).to(self.device)
            self.features_omics2 = torch.FloatTensor(self.adata_omics2.obsm['input_feat'].copy()).to(self.device)

            self.dim_input1 = self.features_omics1.shape[1]
            self.dim_input2 = self.features_omics2.shape[1]
           
            if self.features_omics1.shape[0] > 4e4:
                self.train2_sparse() # Use sparse format and batch training
            else:
                self.train2() 

        if self.n_views >= 3:
            modality = list(self.data.keys())
            self.mlpLayer_list = []
            self.dim_input_list = []
            self.adata_list = []
            self.features_list = []
            
            for m in modality:
                mlpLayer = [128, 64] if m in ['protein'] else [256, 64]
                self.mlpLayer_list.append(mlpLayer)
                
                adata_omics = self.data[m]
                self.adata_list.append(adata_omics)
                
                features_omics = torch.FloatTensor(adata_omics.obsm['input_feat'].copy()).to(self.device)
                self.features_list.append(features_omics)
                
                dim_input = features_omics.shape[1]
                self.dim_input_list.append(dim_input)
    
            self.train_va()
     

    def train2(self):
        self.model = DePassAE(dim_input1=self.dim_input1, dim_input2=self.dim_input2, mlplayer1=self.mlpLayer1,
                           mlplayer2=self.mlpLayer2,dim=self.dim).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.learning_rate)
        self.scheduler = MultiStepLR(self.optimizer, milestones=[80,160], gamma=0.5)
        
        print('[Initializing]')
        print('Graph Construction : Running...')
        adj1 = construct_knn_graph_hnsw(data=self.adata_omics1.obsm['input_feat'].copy(),k=self.K_feature)
        adj2 = construct_knn_graph_hnsw(data=self.adata_omics2.obsm['input_feat'].copy(),k=self.K_feature)
       
        if self.data_type == 'spatial':
            xy = self.adata_omics1.obsm['spatial'].copy()
            adj_shared = construct_knn_graph_hnsw(data=xy,k=self.K_spatial)

        elif self.data_type == 'single_cell':
            adj_shared = adj1 + adj2
            adj_shared.data[adj_shared.data > 1] = 1
        print('Graph Construction : Done!')

        print('Data Enhancement : Running...')
        k_xNeigh = 3
        k_zNeigh = 10 if self.data_type == 'spatial' else 3
        FeatNeigh1 = construct_knn_graph_hnsw(data=self.adata_omics1.obsm['input_feat'].copy(),k=k_xNeigh)
        FeatNeigh2 = construct_knn_graph_hnsw(data=self.adata_omics2.obsm['input_feat'].copy(),k=k_xNeigh)
        SpatialNeigh = adj_shared if self.data_type == 'spatial' else None
        x1_self_enh = get_augment(data=self.features_omics1,SpatialNeigh=SpatialNeigh,FeatNeigh=FeatNeigh1,w_fea=0.1,w_spa=0.1)            
        x2_self_enh = get_augment(data=self.features_omics2,SpatialNeigh=SpatialNeigh,FeatNeigh=FeatNeigh2,w_fea=0.1,w_spa=0.1) 
        print('Data Enhancement : Done!')

        x1_z_enh =  x1_self_enh
        x2_z_enh =  x2_self_enh

        adj1_norm = normalize_adj(adj1).to(self.device)
        adj2_norm = normalize_adj(adj2).to(self.device)
        adj_shared_norm = normalize_adj(adj_shared).to(self.device)
    
        if self.sub_graph_size:
            print('')
            undirected_geo = to_undirected_geo_data(adj_shared,node_index=torch.arange(adj_shared.shape[0]))
            cluster_data = ClusterData(undirected_geo, num_parts=int(np.ceil(undirected_geo.num_nodes / self.sub_graph_size)) * 4,
                                       recursive=False)
            train_loader = ClusterLoader(cluster_data, batch_size=2, shuffle=True)
        
        print('')
        print('[Training]')
        print("Model training starts...")  
        self.model.train()
        for epoch in tqdm(range(0, self.epochs)):
            e1, e2 = self.model.mlp_forward(x1=x1_self_enh, z1=x1_z_enh, x2=x2_self_enh, z2=x2_z_enh)
            if self.batch_training:
                loss_recon1_list = []
                loss_recon2_list = []
                for cur in train_loader:
                    np_index = cur.node_index.numpy()
                    e1_batch = e1[np_index]
                    e2_batch = e2[np_index]
                    adj1_batch = adj1[np_index, :][:, np_index]
                    adj2_batch = adj2[np_index, :][:, np_index]
                    adj_shared_batch = adj_shared[np_index, :][:, np_index]
                    adj1_batch = normalize_adj(adj1_batch).to(self.device)
                    adj2_batch = normalize_adj(adj2_batch).to(self.device)
                    adj_shared_batch = normalize_adj(adj_shared_batch).to(self.device)
                    results = self.model.gcn_forward(e1_batch, e2_batch, adj_shared_batch, adj1_batch, adj2_batch)
                    loss_recon1_ = F.mse_loss(self.features_omics1[np_index], results['emb_recon1'])
                    loss_recon2_ = F.mse_loss(self.features_omics2[np_index], results['emb_recon2'])
                    loss_recon1_list.append(loss_recon1_)
                    loss_recon2_list.append(loss_recon2_)
                loss_recon1 = torch.mean(torch.stack(loss_recon1_list))
                loss_recon2 = torch.mean(torch.stack(loss_recon2_list))
            else:    
                results = self.model.gcn_forward(e1, e2, adj_shared_norm, adj1_norm, adj2_norm)
                loss_recon1 = F.mse_loss(self.features_omics1, results['emb_recon1'])
                loss_recon2 = F.mse_loss(self.features_omics2, results['emb_recon2'])

            loss = loss_recon1 + loss_recon2 
         
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()  

            if epoch > 0 and epoch % self.updata_step == 0:
                with torch.no_grad():
                    self.model.eval()
                    results = self.model.gcn_inference(e1,e2,adj_shared_norm,adj1_norm,adj2_norm)
                    self.model.train()
                    ZNeigh = construct_knn_graph_hnsw(data=results['z'].detach().cpu().numpy(),k=k_zNeigh)
                    x1_z_enh = get_augment(data=self.features_omics1,SpatialNeigh=SpatialNeigh,FeatNeigh=ZNeigh)            
                    x2_z_enh = get_augment(data=self.features_omics2,SpatialNeigh=SpatialNeigh,FeatNeigh=ZNeigh) 
                    adj1 = construct_knn_graph_hnsw(data=e1.detach().cpu().numpy(),k=self.K_feature)
                    adj2 = construct_knn_graph_hnsw(data=e2.detach().cpu().numpy(),k=self.K_feature)
                    adj1_norm = normalize_adj(adj1).to(self.device)
                    adj2_norm = normalize_adj(adj2).to(self.device)
                    if self.data_type == 'single_cell':
                        adj_shared = adj1 + adj2
                        adj_shared.data[adj_shared.data > 1] = 1
                        adj_shared_norm = normalize_adj(adj_shared).to(self.device)

        
        print("Model training finished!\n")
        with torch.no_grad():
            self.model.eval()
            results = self.model.gcn_inference(e1, e2, adj_shared_norm, adj1_norm, adj2_norm)
            ZNeigh = construct_knn_graph_hnsw(data=results['z'].detach().cpu().numpy(),k=k_zNeigh)
            x1_z_enh = get_augment(data=self.features_omics1,SpatialNeigh=SpatialNeigh,FeatNeigh=ZNeigh)             
            x2_z_enh = get_augment(data=self.features_omics2,SpatialNeigh=SpatialNeigh,FeatNeigh=ZNeigh)  
      
        self.embedding = norm_convert(results['z'])
        self.z1 = norm_convert(results['z1'])
        self.z2 = norm_convert(results['z2'])
        self.alpha = results['alpha'].detach().cpu().numpy()
        self.x1_enh = x1_z_enh.detach().cpu().numpy()
        self.x2_enh = x2_z_enh.detach().cpu().numpy()

    def train2_sparse(self):
        self.model = DePassAE(dim_input1=self.dim_input1, dim_input2=self.dim_input2, mlplayer1=self.mlpLayer1,
                           mlplayer2=self.mlpLayer2,dim=self.dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.learning_rate)
        self.scheduler = MultiStepLR(self.optimizer, milestones=[80,160], gamma=0.5)

        print('[Initializing]')
        print('Graph Construction : Running...')
        adj1 = construct_knn_graph_hnsw_sparse(self.adata_omics1.obsm['input_feat'].copy(), k=self.K_feature)
        adj2 = construct_knn_graph_hnsw_sparse(self.adata_omics2.obsm['input_feat'].copy(), k=self.K_feature)
        if self.data_type == 'spatial':
            adj_shared = construct_knn_graph_hnsw_sparse(self.adata_omics1.obsm['spatial'].copy(), k=self.K_spatial)
        elif self.data_type == 'single_cell':
            adj_shared = adj1 + adj2
            adj_shared.data[adj_shared.data > 1] = 1
        print('Graph Construction : Done!')

        print('Data Enhancement : Running...')
        k_xNeigh = 6
        # k_zNeigh = int(self.K_feature/2) if self.data_type == 'spatial' else 6
        k_zNeigh = 20 if self.data_type == 'spatial' else 6
        FeatNeigh1 = construct_knn_graph_hnsw_sparse(self.adata_omics1.obsm['input_feat'].copy(), k=k_xNeigh)
        FeatNeigh2 = construct_knn_graph_hnsw_sparse(self.adata_omics2.obsm['input_feat'].copy(), k=k_xNeigh)
        SpatialNeigh = adj_shared if self.data_type == 'spatial' else None
        x1_self_enh = augment_data_lc(self.features_omics1, SpatialNeigh=SpatialNeigh,FeatNeigh=FeatNeigh1,w_fea=0.1,w_spa=0.1)
        x2_self_enh = augment_data_lc(self.features_omics2, SpatialNeigh=SpatialNeigh,FeatNeigh=FeatNeigh2,w_fea=0.1,w_spa=0.1)
        x1_z_enh = x1_self_enh 
        x2_z_enh = x2_self_enh 
        print('Data Enhancement : Done!')

        adj1_norm = normalize_adj_scr(adj1).to(self.device)
        adj2_norm = normalize_adj_scr(adj2).to(self.device)
        adj_shared_norm = normalize_adj_scr(adj_shared).to(self.device)

        # batch_training
        undirected_geo = to_undirected_geo_data(adj_shared,node_index=torch.arange(adj_shared.shape[0]))
        cluster_data = ClusterData(undirected_geo, num_parts=int(np.ceil(undirected_geo.num_nodes / self.sub_graph_size)) * 4,
                                       recursive=False)
        train_loader = ClusterLoader(cluster_data, batch_size=2, shuffle=True)

        print('')
        print('[Training]')
        print("Model training starts...")
        self.model.train()
        for epoch in tqdm(range(0, self.epochs)):
            e1, e2 = self.model.mlp_forward(x1=x1_self_enh, z1=x1_z_enh, x2=x2_self_enh, z2=x2_z_enh)
            loss_recon1_list = []
            loss_recon2_list = []
            for cur in train_loader:
                np_index = cur.node_index.numpy()
                e1_batch = e1[np_index]
                e2_batch = e2[np_index]
                adj1_batch = sparse_indexing(adj1, np_index)
                adj2_batch = sparse_indexing(adj2, np_index)
                adj_shared_batch = sparse_indexing(adj_shared, np_index)
                adj1_batch = normalize_adj_scr(adj1_batch).to(self.device)
                adj2_batch = normalize_adj_scr(adj2_batch).to(self.device)
                adj_shared_batch = normalize_adj_scr(adj_shared_batch).to(self.device)
                results = self.model.gcn_forward(e1_batch, e2_batch, adj_shared_batch, adj1_batch, adj2_batch)
                loss_recon1_ = F.mse_loss(self.features_omics1[np_index], results['emb_recon1'])
                loss_recon2_ = F.mse_loss(self.features_omics2[np_index], results['emb_recon2'])
                loss_recon1_list.append(loss_recon1_)
                loss_recon2_list.append(loss_recon2_)

            loss_recon1 = torch.mean(torch.stack(loss_recon1_list))
            loss_recon2 = torch.mean(torch.stack(loss_recon2_list))
            loss = loss_recon1 + loss_recon2 
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            if epoch>0 and epoch % self.updata_step == 0:
                with torch.no_grad():
                    self.model.eval()
                    results = self.model.gcn_inference(e1,e2,adj_shared_norm,adj1_norm,adj2_norm)
                    self.model.train()
                    ZNeigh = construct_knn_graph_hnsw_sparse(results['z'].detach().cpu().numpy(), k=k_zNeigh) 
                    x1_z_enh = augment_data_lc(self.features_omics1,SpatialNeigh=SpatialNeigh,FeatNeigh=ZNeigh)   
                    x2_z_enh = augment_data_lc(self.features_omics2,SpatialNeigh=SpatialNeigh,FeatNeigh=ZNeigh) 
                    adj1 = construct_knn_graph_hnsw_sparse(e1.detach().cpu().numpy(), k=self.K_feature)
                    adj2 = construct_knn_graph_hnsw_sparse(e2.detach().cpu().numpy(), k=self.K_feature)
                    adj1_norm = normalize_adj_scr(adj1).to(self.device)
                    adj2_norm = normalize_adj_scr(adj2).to(self.device)
                    
                    if self.data_type == 'single_cell':
                        adj_shared = adj1 + adj2
                        adj_shared.data[adj_shared.data > 1] = 1
                        adj_shared_norm = normalize_adj_scr(adj_shared).to(self.device)

        print("Model training finished!\n")
        with torch.no_grad():
            self.model.eval()
            results = self.model.gcn_inference(e1, e2, adj_shared_norm, adj1_norm, adj2_norm)
            ZNeigh = construct_knn_graph_hnsw_sparse(results['z'].detach().cpu().numpy(), k=k_zNeigh) 
            x1_z_enh  = augment_data_lc(self.features_omics1,SpatialNeigh=SpatialNeigh,FeatNeigh=ZNeigh)   
            x2_z_enh  = augment_data_lc(self.features_omics2,SpatialNeigh=SpatialNeigh,FeatNeigh=ZNeigh) 
     
        self.embedding = norm_convert(results['z'])
        self.z1 = norm_convert(results['z1'])
        self.z2 = norm_convert(results['z2'])
        self.alpha = results['alpha'].detach().cpu().numpy()
        self.x1_enh  = x1_z_enh.detach().cpu().numpy()
        self.x2_enh  = x2_z_enh.detach().cpu().numpy()
    
    def train_va(self):
        self.model = DePassAE_va(dim_input_list=self.dim_input_list, mlplayer_list=self.mlpLayer_list, dim=self.dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = MultiStepLR(self.optimizer, milestones=[80, 160], gamma=0.5)

        print('[Initializing]')
        print('Graph Construction : Running...')
        adj_list = []
        for adata in self.adata_list:
            adj = construct_knn_graph_hnsw(data=adata.obsm['input_feat'].copy(), k=self.K_feature)
            adj_list.append(adj)

        if self.data_type == 'spatial':
            xy = self.adata_list[0].obsm['spatial'].copy()
            adj_shared = construct_knn_graph_hnsw(data=xy, k=self.K_spatial)
        elif self.data_type == 'single_cell':
            adj_shared = sum(adj_list)
            adj_shared.data[adj_shared.data > 1] = 1
        print('Graph Construction : Done!')

        print('Data Enhancement : Running...')
        k_xNeigh = 3
        k_zNeigh = int(self.K_feature/2) if self.data_type == 'spatial' else 3
        FeatNeigh_list = []
        for adata in self.adata_list:
            FeatNeigh = construct_knn_graph_hnsw(data=adata.obsm['input_feat'].copy(), k=k_xNeigh)
            FeatNeigh_list.append(FeatNeigh)

        SpatialNeigh = adj_shared if self.data_type == 'spatial' else None
        x_self_enh_list = []
        for i in range(len(self.features_list)):
            x_self_enh = get_augment(data=self.features_list[i], SpatialNeigh=SpatialNeigh, FeatNeigh=FeatNeigh_list[i], w_fea=0.1, w_spa=0.1)
            x_self_enh_list.append(x_self_enh)
        x_z_enh_list =  x_self_enh_list 

        print('Data Enhancement : Done!')

        adj_norm_list = []
        for adj in adj_list:
            adj_norm = normalize_adj(adj).to(self.device)
            adj_norm_list.append(adj_norm)
        adj_shared_norm = normalize_adj(adj_shared).to(self.device)

        print('')
        print('[Training]')
        print("Model training starts...")
        self.model.train()
        for epoch in tqdm(range(0, self.epochs)):
            e_list = self.model.mlp_forward(xs=x_self_enh_list, zs=x_z_enh_list)
            results = self.model.gcn_forward(e_list, adj_shared_norm, adj_norm_list)
            loss_recon_list = []
            for i in range(len(self.features_list)):
                loss_recon = F.mse_loss(self.features_list[i], results['emb_recons'][i])
                loss_recon_list.append(loss_recon)

            loss = sum(loss_recon_list)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            if epoch > 0 and epoch % self.updata_step == 0:
                with torch.no_grad():
                    self.model.eval()
                    results = self.model.gcn_inference(e_list, adj_shared_norm, adj_norm_list)
                    self.model.train()
                    ZNeigh = construct_knn_graph_hnsw(data=results['z'].detach().cpu().numpy(), k=k_zNeigh)

                    x_z_enh_list = []
                    for i in range(len(self.features_list)):
                        x_z_enh = get_augment(data=self.features_list[i], SpatialNeigh=SpatialNeigh, FeatNeigh=ZNeigh)
                        x_z_enh_list.append(x_z_enh)

                    for i in range(len(e_list)):
                        adj = construct_knn_graph_hnsw(data=e_list[i].detach().cpu().numpy(), k=self.K_feature)
                        adj_norm_list[i] = normalize_adj(adj).to(self.device)

                    if self.data_type == 'single_cell':
                        adj_shared = sum(adj_list)
                        adj_shared.data[adj_shared.data > 1] = 1
                        adj_shared_norm = normalize_adj(adj_shared).to(self.device)

        print("Model training finished!\n")
        with torch.no_grad():
            self.model.eval()
            results = self.model.gcn_inference(e_list, adj_shared_norm, adj_norm_list)
            ZNeigh = construct_knn_graph_hnsw(data=results['z'].detach().cpu().numpy(), k=k_zNeigh)
            x_z_enh_list = []
            for i in range(len(self.features_list)):
                x_z_enh = get_augment(data=self.features_list[i], SpatialNeigh=SpatialNeigh, FeatNeigh=ZNeigh)
                x_z_enh_list.append(x_z_enh)

        self.embedding = norm_convert(results['z'])
        self.alpha = results['alpha'].detach().cpu().numpy()
        for i, x in enumerate(x_z_enh_list):
            setattr(self, f'x{i+1}_enh', x.detach().cpu().numpy())
    


import torch
from scipy.sparse import coo_matrix

def get_augment(data=None,FeatNeigh=None, SpatialNeigh=None, w = 0.01, w_fea=0.25,w_spa=0.25):
        FeatNeigh = FeatNeigh.to(data.device)
        if SpatialNeigh is None:
            enhanced_data = data + w * FeatNeigh @ data 
        if SpatialNeigh is not None:
            SpatialNeigh = SpatialNeigh.to(data.device)
            enhanced_data = data + w_fea * FeatNeigh @ data + w_spa * SpatialNeigh @ data      
        return enhanced_data

def toTensor_sparse(data,device):
        num_samples, _ = data.shape
        if not isinstance(data, coo_matrix):
            data = data.tocoo()
        indices = torch.tensor([data.row, data.col], dtype=torch.long, device=device)
        values = torch.tensor(data.data, dtype=torch.float32, device=device)
        data = torch.sparse.FloatTensor(indices, values, (num_samples, num_samples))
        return data

def augment_data_lc(data, FeatNeigh=None, SpatialNeigh=None, w=0.01, w_fea=0.25, w_spa=0.25):
    FeatNeigh = FeatNeigh.to(data.device)
    if SpatialNeigh is None:
                enhanced_data = data + w * FeatNeigh @ data 
        
    if SpatialNeigh is not None:
       SpatialNeigh = SpatialNeigh.to(data.device)
       enhanced_data = data.clone()
      
       feat_contribution = w_fea * torch.sparse.mm(FeatNeigh, data)
       enhanced_data += feat_contribution
       del feat_contribution  
       if torch.cuda.is_available():
           torch.cuda.empty_cache()  
       
       spa_contribution = w_spa * torch.sparse.mm(SpatialNeigh, data)
       enhanced_data += spa_contribution
       del spa_contribution  
       if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return enhanced_data

def norm_convert(emb):
    emb = F.normalize(emb, p=2, eps=1e-12, dim=1)
    return emb.detach().cpu().numpy()
