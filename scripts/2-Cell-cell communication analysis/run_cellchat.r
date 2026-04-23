# ===============================
# 1. Load libraries and environment
# ===============================
library(Seurat)
library(CellChat)
library(reticulate)
library(ComplexHeatmap)
library(circlize)
library(ggsci)
library(cowplot)
library(patchwork)
library(RColorBrewer)
library(ggplot2)
library(dplyr)
library(readr)

packageVersion("CellChat")
use_python("/home/jyx2/miniconda3/envs/ccc/bin/python")
dir.create("./raw", showWarnings = FALSE)


# ===============================
# 2. Load AnnData and preprocess expression matrix
# ===============================
ad <- import("anndata")
adata <- ad$read_h5ad("/home/jyx2/DePass-main/1-R1/C-cellchat/rna_raw.h5ad")
# adata <- ad$read_h5ad("/home/jyx2/DePass-main/1-R1/C-cellchat/rna_enhanced.h5ad")


expr <- t(as.matrix(adata$X))  
rownames(expr) <- adata$var_names$to_list()  
colnames(expr) <- adata$obs_names$to_list()  

meta <- adata$obs
rownames(meta) <- adata$obs_names$to_list()



# ===============================
# 3. Build and run CellChat 
# ===============================
cellchat <- createCellChat(object = expr, meta = meta, group.by = "DePass25")
cellchat@DB <- CellChatDB.human
cellchat <- subsetData(cellchat)
cellchat <- identifyOverExpressedGenes(cellchat, do.fast = FALSE)
cellchat <- identifyOverExpressedInteractions(cellchat)
cellchat <- computeCommunProb(cellchat, type = "truncatedMean", trim = 0.1)
cellchat <- computeCommunProbPathway(cellchat)
cellchat <- aggregateNet(cellchat)



# ===============================
# 4. Export communication results
# ===============================
df_net <- subsetCommunication(cellchat, slot.name = "net") %>%
  select(source, target, ligand, receptor, prob, pval) %>%
  mutate(lr_pair = paste(ligand, receptor, sep = "_"))
write_csv(df_net, "raw/df_net_raw.csv")  



# ===============================
# 5. Bubble plot for cluster 12 and 17
# ===============================
p_obj <- netVisual_bubble(cellchat, sources.use = c("12","17"), targets.use = c("12","17"), return.data = TRUE)
df_all <- p_obj$communication

groupA <- '12'
groupB <- '17'

df <- subset(df_all, source != target)
df <- df[order(-df$prob), ]

df$cell_pair <- paste0(df$source, "->", df$target)
df$lr_label <- df$interaction_name_2  
df$cell_pair <- factor(df$cell_pair)
df$lr_label <- factor(df$lr_label)

write.table(df, 
            file = "./raw/cell_communication_data_1217.txt", 
            sep = ",", quote = FALSE, row.names = FALSE)

df <- head(df, 12)

p2 <- ggplot(df, aes(x = cell_pair, y = lr_label)) +
  geom_point(aes(color = prob, size = as.factor(pval)), alpha=0.9) +
  scale_color_gradientn(colours = c("#80B8E8","#4086C2","#0A4A8F"),
                       name = "Commun. Prob.") +
  scale_size_manual(values=c("3"=8), labels=c("p<0.01"), name="p-value") +
  labs(title = "12 ↔ 17 Communication", x="", y="") +
  theme_classic() +
  theme(plot.title = element_text(hjust = 0.5, size = 20, face = "bold"),
        axis.text.x = element_text(angle = 45, hjust = 1, size = 14),
        axis.text.y = element_text(size = 14),
        panel.grid.major = element_line(color = "gray92", linewidth = 0.3),
        legend.key.size = unit(0.8, "cm"),
        legend.text = element_text(size = 14),
        legend.title = element_text(size = 14, face = "bold"))

print(p2)

pdf("./raw/CCC_Bubble_tumor.pdf", width=8, height=9)
print(p2)
dev.off()



# ===============================
# 6. Bubble plot for cluster 21 and 22
# ===============================
p_obj <- netVisual_bubble(cellchat, sources.use = c("21","22"), targets.use = c("21","22"), return.data = TRUE)
df_all <- p_obj$communication

groupA <- '21'
groupB <- '22'

df <- subset(df_all, source != target)
df <- df[order(-df$prob), ]

df$cell_pair <- paste0(df$source, "->", df$target)
df$lr_label <- df$interaction_name_2

df$cell_pair <- factor(df$cell_pair)
df$lr_label <- factor(df$lr_label)

write.table(df, 
            file = "./raw/cell_communication_data_2122.txt", 
            sep = ",", quote = FALSE, row.names = FALSE)

df <- head(df, 12)

p2 <- ggplot(df, aes(x = cell_pair, y = lr_label)) +
  geom_point(aes(color = prob, size = as.factor(pval)), alpha = 0.92) +
  scale_color_gradientn(colours = c("#80B8E8","#4086C2","#0A4A8F"), name = "Commun. Prob.") +
  scale_size_manual(values = c("3" = 9), labels = c("p < 0.01"), name = "Significance") +
  labs(title = paste0(groupA, " ↔ ", groupB, " Communication"), x = "", y = "") +
  theme_classic() +
  theme(plot.title = element_text(hjust = 0.5, size = 20, face = "bold"),
        axis.text.x = element_text(angle = 45, hjust = 1, size = 14),
        axis.text.y = element_text(size = 14),
        panel.grid.major = element_line(color = "gray92", linewidth = 0.3),
        legend.key.size = unit(0.8, "cm"),
        legend.text = element_text(size = 14),
        legend.title = element_text(size = 14, face = "bold"))

print(p2)

pdf("./raw/CCC_Bubble_2122.pdf", width = 8, height = 9)
print(p2)
dev.off()