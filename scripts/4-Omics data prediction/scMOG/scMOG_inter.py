
# The source code is borrowed from [GitHub: https://github.com/GaoLabXDU/scMOG]

import argparse
import itertools
import logging
import os
import random
import sys
import time
from typing import *
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torch.utils.data as Data
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../..'))
sys.path.append(project_root)
import scanpy as sc
from methods.utils import *
from sklearn.model_selection import train_test_split
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scMOG")
assert os.path.isdir(SRC_DIR)
sys.path.append(SRC_DIR)
MODELS_DIR = os.path.join(SRC_DIR, "models")
assert os.path.isdir(MODELS_DIR)
sys.path.append(MODELS_DIR)
import both_GAN_1
import lossfunction
import methods.utils as utils
from pytorchtools import EarlyStopping
logging.basicConfig(level=logging.INFO)
SAVEFIG_DPI = 1200

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--hidden", type=int, nargs="*", default=[16])
    parser.add_argument("--lr", "-l", type=float, default=[0.0001], nargs="*")
    parser.add_argument("--batchsize", "-b", type=int, nargs="*", default=[512])
    parser.add_argument("--seed", type=int, nargs="*", default=1105)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--ext", type=str, choices=["png", "pdf", "jpg"], default="pdf")
    parser.add_argument("--b1", type=float, default=0.5)
    parser.add_argument("--b2", type=float, default=0.999)
    parser.add_argument("--clip_value", type=float, default=0.01)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--train_size', type=float, default=0.8)
    parser.add_argument('--test_size', type=float, default=0.1)
    parser.add_argument('--RNA_path_train', default='/dataset/rna_pro/inter/dataset12_dataset23/dataset12/adata_RNA.h5ad')
    parser.add_argument('--Pro_path_train', default='/dataset/rna_pro/inter/dataset12_dataset23/dataset12/adata_ADT.h5ad')
    parser.add_argument('--RNA_path_test', default='/dataset/rna_pro/inter/dataset12_dataset23/dataset23/adata_RNA.h5ad')
    parser.add_argument('--Pro_path_test', default='/dataset/rna_pro/inter/dataset12_dataset23/dataset23/adata_ADT.h5ad')
    parser.add_argument('--method_flag', default='scMOG')
    parser.add_argument('--mode', default='sc')
   
    return parser

def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def plot_loss_history(history1,history2,history3,fname: str):
    fig, ax = plt.subplots(dpi=300)
    ax.plot(np.arange(len(history1)), history1, label="Train_G")
    if len(history2):
        ax.plot(np.arange(len(history2)), history2, label="Train_D")
    if len(history3):
        ax.plot(np.arange(len(history3)), history3, label="Test_G")
    ax.legend()
    ax.set(xlabel="Epoch", ylabel="Loss")
    fig.savefig(fname)
    return fig

def plot_auroc(truth, preds, title_prefix: str = "Receiver operating characteristic", fname: str = ""):
    truth = truth.cpu().numpy().flatten()
    preds = preds.cpu().numpy().flatten()
    fpr, tpr, _thresholds = metrics.roc_curve(truth, preds)
    auc = metrics.auc(fpr, tpr)
    logging.info(f"Found AUROC of {auc:.4f}")

    fig, ax = plt.subplots(dpi=300, figsize=(7, 5))
    ax.plot(fpr, tpr)
    ax.set(xlim=(0, 1.0), ylim=(0.0, 1.05), xlabel="False positive rate", ylabel="True positive rate",
           title=f"{title_prefix} (AUROC={auc:.2f})")
    if fname:
        fig.savefig(fname, dpi=SAVEFIG_DPI, bbox_inches="tight")
    return fig

def plot_prc(truth, preds, title_prefix: str = "Receiver operating characteristic", fname: str = ""):
    truth = truth.cpu().numpy().flatten()
    preds = preds.cpu().numpy().flatten()
    precision, recall, _thresholds = metrics.precision_recall_curve(truth, preds)
    auc = metrics.auc(recall,precision)
    logging.info(f"Found AUPRC of {auc:.4f}")

    fig, ax = plt.subplots(dpi=300, figsize=(7, 5))
    ax.plot(recall, precision)
    ax.set(xlim=(0, 1.0), ylim=(0.0, 1.05), xlabel="recall", ylabel="precision",
           title=f"{title_prefix} (PRC={auc:.2f})")
    if fname:
        fig.savefig(fname, dpi=SAVEFIG_DPI, bbox_inches="tight")
    return fig

def rmse_value(truth, preds):
    truth = truth.cpu().numpy().flatten()
    preds = preds.cpu().numpy().flatten()
    rmse=np.sqrt(metrics.mean_squared_error(truth, preds))
    logging.info(f"Found RMSE of {rmse:.4f}")

def plot_scatter_with_r(x: Union[np.ndarray, scipy.sparse.csr_matrix], y: Union[np.ndarray, scipy.sparse.csr_matrix],
                        color=None, subset: int = 0, logscale: bool = False, density_heatmap: bool = False,
                        density_dpi: int = 150, density_logstretch: int = 1000, title: str = "",
                        xlabel: str = "Original norm counts", ylabel: str = "Inferred norm counts",
                        xlim: Tuple[int, int] = None, ylim: Tuple[int, int] = None, one_to_one: bool = False,
                        corr_func: Callable = scipy.stats.pearsonr, figsize: Tuple[float, float] = (7, 5),
                        fname: str = "", ax=None):
    assert x.shape == y.shape
    if color is not None:
        assert color.size == x.size
    if one_to_one and (xlim is not None or ylim is not None):
        assert xlim == ylim
    if xlim:
        keep_idx = utils.ensure_arr((x >= xlim[0]).multiply(x <= xlim[1]))
        x = utils.ensure_arr(x[keep_idx])
        y = utils.ensure_arr(y[keep_idx])
    if ylim:
        keep_idx = utils.ensure_arr((y >= ylim[0]).multiply(x <= xlim[1]))
        x = utils.ensure_arr(x[keep_idx])
        y = utils.ensure_arr(y[keep_idx])
    assert x.shape == y.shape
    
    if subset > 0 and subset < x.size:
        logging.info(f"Subsetting to {subset} points")
        random.seed(1234)
        indices = np.unravel_index(np.array(random.sample(range(np.product(x.shape)), k=subset)), shape=x.shape)
        x = utils.ensure_arr(x[indices])
        y = utils.ensure_arr(y[indices])
        if isinstance(color, (tuple, list, np.ndarray)):
            color = np.array([color[i] for i in indices])

    if logscale:
        x = np.log1p(x.cpu())
        y = np.log1p(y.cpu())

    x = x.cpu().numpy().flatten()
    y = y.cpu().numpy().flatten()
    assert not np.any(np.isnan(x))
    assert not np.any(np.isnan(y))

    pearson_r, pearson_p = scipy.stats.pearsonr(x, y)
    logging.info(f"Found pearson's correlation/p of {pearson_r:.4f}/{pearson_p:.4g}")
    spearman_corr, spearman_p = scipy.stats.spearmanr(x, y)
    logging.info(f"Found spearman's collelation/p of {spearman_corr:.4f}/{spearman_p:.4g}")

    if ax is None:
        fig = plt.figure(dpi=300, figsize=figsize)
        if density_heatmap:
            ax = fig.add_subplot(1, 1, 1, projection="scatter_density")
        else:
            ax = fig.add_subplot(1, 1, 1)
    else:
        fig = None

    if density_heatmap:
        norm = None
        if density_logstretch:
            norm = ImageNormalize(vmin=0, vmax=100, stretch=LogStretch(a=density_logstretch))
        ax.scatter_density(x, y, dpi=density_dpi, norm=norm, color="tab:blue")
    else:
        ax.scatter(x, y, alpha=0.2, c=color)

    if one_to_one:
        unit = np.linspace(*ax.get_xlim())
        ax.plot(unit, unit, linestyle="--", alpha=0.5, label="$y=x$", color="grey")
        ax.legend()
    ax.set(xlabel=xlabel + (" (log)" if logscale else ""), ylabel=ylabel + (" (log)" if logscale else ""),
           title=(title + f" ($r={pearson_r:.2f}$)").strip())
    if xlim:
        ax.set(xlim=xlim)
    if ylim:
        ax.set(ylim=ylim)

    if fig is not None and fname:
        fig.savefig(fname, dpi=SAVEFIG_DPI, bbox_inches="tight")

    return fig

def main():
    parser = build_parser()
   
    args = parser.parse_args()

    logger = logging.getLogger()
    fh = logging.FileHandler("training.log", "w")
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    if torch.cuda.is_available():
        logging.info(f"PyTorch CUDA version: {torch.version.cuda}")
    for arg in vars(args):
        logging.info(f"Parameter {arg}: {getattr(args, arg)}")

    scRNA_adata_train = sc.read_h5ad(args.RNA_path_train) 
    scP_adata_train =   sc.read_h5ad(args.Pro_path_train)
    scRNA_adata_test =  sc.read_h5ad(args.RNA_path_test)
    scP_adata_test =    sc.read_h5ad(args.Pro_path_test)

    scRNA_adata_train,scRNA_adata_test = select_feature_inter(scRNA_adata_train,scRNA_adata_test,hvg_num=3000)
    scP_adata_train,scP_adata_test = select_feature_inter(scP_adata_train,scP_adata_test,hvg_num=False)
 
 
    train_index, val_index = train_test_split(scRNA_adata_train.obs.index, test_size=args.test_size, random_state=args.seed + args.repeat)
    train_rna = scRNA_adata_train[train_index]
    train_pro = scP_adata_train[train_index]
    val_rna = scRNA_adata_train[val_index]
    val_pro = scP_adata_train[val_index]

    test_rna = scRNA_adata_test
    test_pro = scP_adata_test

    train_rna = preprocess_rna_inter(train_rna,norm=False)
    val_rna = preprocess_rna_inter(val_rna,norm=False)
    test_rna = preprocess_rna_inter(test_rna,norm=False)
    train_pro = preprocess_pro_inter(train_pro, norm=False)
    val_pro = preprocess_pro_inter(val_pro, norm=False)
    test_pro = preprocess_pro_inter(test_pro, norm=False)

    sc_rna_train_dataset=train_rna
    sc_pro_train_dataset=train_pro
    sc_rna_test_dataset=val_rna
    sc_pro_test_dataset=val_pro
    true_test_rna = test_rna
    true_test_pro = test_pro

    cuda = True if torch.cuda.is_available() else False
    device_ids = range(torch.cuda.device_count())

    param_combos = list(itertools.product(args.hidden, args.lr, [args.seed + args.repeat]))
    for h_dim, lr, rand_seed in param_combos:

        GeneratorProtein = both_GAN_1.GeneratorProtein(hidden_dim=h_dim,
                                           input_dim1=sc_rna_test_dataset.X.shape[1],
                                           input_dim2=sc_pro_test_dataset.X.shape[1],
                                           final_activations2=nn.Identity(),
                                           flat_mode=True,
                                           seed=rand_seed,
                                           )

        DiscriminatorProtein = both_GAN_1.DiscriminatorProtein(input_dim=sc_pro_train_dataset.X.shape[1],seed=rand_seed)

        loss_rna = lossfunction.loss
        loss_protein=nn.MSELoss()

        def loss_D(fake,real,Discriminator):
            loss2_1 = -torch.mean(Discriminator(real))
            if isinstance(fake, tuple):
                loss2_2 = torch.mean(Discriminator(fake[0].detach()))
            else:
                loss2_2 = torch.mean(Discriminator(fake.detach()))
            loss2 = loss2_1 + loss2_2
            return loss2

        def loss_rna_G(fake,Discriminator):
            loss1 =-torch.mean(Discriminator(fake[0]))
            return loss1

        def loss_atac_G(fake,Discriminator):
            loss1 = -torch.mean(Discriminator(fake))
            return loss1

        if len(device_ids) > 1:
            GeneratorProtein = torch.nn.DataParallel(GeneratorProtein)
            DiscriminatorProtein = torch.nn.DataParallel(DiscriminatorProtein)

        optimizer_protein_1 = torch.optim.Adam(GeneratorProtein.parameters(), lr=lr, betas=(args.b1, args.b2))
        optimizer_protein = torch.optim.RMSprop(GeneratorProtein.parameters(), lr=lr)
        optimizer_D_protein = torch.optim.RMSprop(DiscriminatorProtein.parameters(), lr=lr)

        def pretrain_epoch(train_iter,generator,discriminator,updaterG,updaterD,lossG_history,lossD_history):
            generator.train()
            discriminator.train()
            train_losses=[]
            trainD_losses=[]
            for i, (x,y) in enumerate(train_iter):
                if cuda:
                    x = x.cuda()
                    y = y.cuda()
                updaterD.zero_grad()
                y_fake = generator(x)
                loss2=loss_D(y_fake,y,discriminator)
                loss2.backward()
                updaterD.step()
                trainD_losses.append(loss2.item())

                for p in discriminator.parameters():
                    p.data.clamp_(-args.clip_value, args.clip_value)

                if i % 5 == 0:
                    updaterG.zero_grad()
                    y_hat =generator(x)
                    if isinstance(y_hat, tuple):
                        loss1=loss_rna_G(y_hat,discriminator)
                    else:
                        loss1 = loss_atac_G(y_hat,discriminator)
                    loss1.backward()
                    updaterG.step()
                    train_losses.append(loss1.item())

            train_loss = np.average(train_losses[:-1])
            trainD_loss = np.average(trainD_losses[:-1])
            logging.info(f"lossG: {train_loss}")
            logging.info(f"lossD: {trainD_loss}")
            lossG_history.append(train_loss)
            lossD_history.append(trainD_loss)

            return lossG_history, lossD_history

        def training_epoch(train_iter, generator, updaterG,lossG_history):
            generator.train()
            train_losses = []
            for i, (x,y) in enumerate(train_iter):
                if cuda:
                    x = x.cuda()
                    y = y.cuda()
                updaterG.zero_grad()
                y_hat = generator(x)
                if isinstance(y_hat, tuple):
                    loss=loss_rna(preds=y_hat[0],theta=y_hat[1],truth=y)
                else:
                    loss = loss_protein(y_hat,y)
                loss.backward()
                updaterG.step()
                train_losses.append(loss.item())
            train_loss = np.average(train_losses[:-1])
            logging.info(f"AEloss: {train_loss}")
            lossG_history.append(train_loss)
            return lossG_history

        def test_epoch(generator,discriminator,test_iter,lossG_test_history):
            generator.eval()
            if discriminator:
                discriminator.eval()
            valid_losses = []
            with torch.no_grad():
                for (x,y)in test_iter:
                    if cuda:
                        x= x.cuda()
                        y= y.cuda()
                    y_hat =generator(x)
                    if discriminator:
                        if isinstance(y_hat, tuple):
                            loss=loss_rna_G(y_hat,discriminator)
                        else:
                            loss= loss_atac_G(y_hat,discriminator)
                    else:
                        if isinstance(y_hat, tuple):
                            loss = loss_rna(preds=y_hat[0], theta=y_hat[1], truth=y)
                        else:
                            loss = loss_protein(y_hat, y)
                    valid_losses.append(loss.item())

            valid_loss = np.average(valid_losses[:-1])
            logging.info(f"loss_test: {valid_loss}")
            lossG_test_history.append(valid_loss)
            return lossG_test_history,valid_loss

        def predict_protein(truth,generator,truth_iter):
            logging.info("....................................Evaluating protein")

            def predict1(generator,truth_iter):
                generator.eval()
                first = 1
                for x in truth_iter:
                    if cuda:
                        x = x.cuda()
                    with torch.no_grad():
                        y_pred = generator(x)
                        if first == 1:
                            ret = y_pred
                            first = 0
                        else:
                            ret = torch.cat((ret, y_pred), 0)
                return ret

            sc_rna_protein_truth_preds = predict1(generator,truth_iter)

            return sc_rna_protein_truth_preds

        def train(generator,discriminator,num_epochs, train_iter,test_iter,truth_iter,truth,updaterG,updaterD,ISRNA):
            lossG_history = []
            lossD_history = []
            lossG_test_history = []
            sc_pro_truth_preds = None

            early_stopping = EarlyStopping(patience=7,verbose=True)
            for epoch in range(num_epochs):
                logging.info(f"....................................................this is epoch: {epoch}")
                if discriminator:
                    lossG_history,lossD_history=pretrain_epoch(train_iter,generator,discriminator,updaterG,updaterD,lossG_history,lossD_history)
                    if ((epoch + 1) %7== 0):
                        sc_pro_truth_preds = predict_protein(truth, generator, truth_iter)
                else:
                    lossG_history=training_epoch(train_iter,generator,updaterG,lossG_history)
                    if ((epoch + 1) % 5== 0):
                        sc_pro_truth_preds = predict_protein(truth, generator, truth_iter)

                if test_iter:
                    lossG_test_history,lossG_test=test_epoch(generator,discriminator,test_iter,lossG_test_history)

                early_stopping(lossG_test, generator)

                if early_stopping.early_stop:
                    logging.info("early stopping")
                    sc_pro_truth_preds = predict_protein(truth, generator, truth_iter)
                    break

            return lossG_history, lossD_history,lossG_test_history, sc_pro_truth_preds

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sc_rna_train = torch.from_numpy(sc_rna_train_dataset.X).float().to(device)
        sc_pro_train = torch.from_numpy(sc_pro_train_dataset.X).float().to(device)
        sc_rna_test = torch.from_numpy(sc_rna_test_dataset.X).float().to(device)
        sc_pro_test = torch.from_numpy(sc_pro_test_dataset.X).float().to(device)
        sc_rna_truth = torch.from_numpy(true_test_rna.X).float().to(device)
        sc_pro_truth = torch.from_numpy(true_test_pro.X).float().to(device)
        
        start_time = time.time()
        train_dataset1= Data.TensorDataset(sc_rna_train, sc_pro_train)
        train_iter1=torch.utils.data.DataLoader(dataset=train_dataset1,batch_size=256,shuffle=True)

        test_dataset1 = Data.TensorDataset(sc_rna_test, sc_pro_test)
        test_iter1 = torch.utils.data.DataLoader(dataset=test_dataset1, batch_size=128)

        truth_iter_rna = torch.utils.data.DataLoader(dataset=sc_rna_truth, batch_size=64)
        
        logging.info("...............................pretraining RNA -> protein")
        loss1_history, loss2_history, loss1_test_history, _ = train(generator=GeneratorProtein,
                                                                 discriminator=DiscriminatorProtein, num_epochs=200,
                                                                 train_iter=train_iter1,
                                                                 test_iter=test_iter1, truth_iter=truth_iter_rna,
                                                                 truth=sc_pro_truth,
                                                                 updaterG=optimizer_protein,
                                                                 updaterD=optimizer_D_protein, ISRNA=False)

        logging.info("........................................................................................................................................................")
        logging.info("training RNA -> protein")
        loss1_history, loss2_history, loss1_test_history, sc_pro_truth_preds = train(generator=GeneratorProtein,
                                                                 discriminator=None, num_epochs=200,
                                                                 train_iter=train_iter1,
                                                                 test_iter=test_iter1, truth_iter=truth_iter_rna,
                                                                 truth=sc_pro_truth,
                                                                 updaterG=optimizer_protein_1,
                                                                 updaterD=None, ISRNA=False)

        test_imputed_pro = sc_pro_truth_preds.cpu().numpy()
        true_pro_adata = true_test_pro
        pred_pro_adata = sc.AnnData(
            X = test_imputed_pro,
            obs=true_pro_adata.obs.copy(),
            var=pd.DataFrame(index=true_pro_adata.var.index))
    
        saveH5adResults(args, pred_pro_adata, true_pro_adata)
        saveResults(
            args, test_imputed_pro, true_pro_adata.X, 
            true_pro_adata.var.index.tolist(), start_time, 
            obs_names=true_pro_adata.obs.index.tolist()
        )
  
if __name__ == "__main__":
    main()