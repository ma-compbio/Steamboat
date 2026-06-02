library(Seurat)
library(CellChat)
library(Matrix)

future::plan("multisession", workers = 8)

expr<- read.csv("G:/data/slidetags/HumanTonsil_expression.csv.gz", row.names = 1)
metadata <- read.csv("G:/data/slidetags/HumanTonsil_metadata.csv", row.names = 1)
spatial <- read.csv("G:/data/slidetags/HumanTonsil_spatial.csv", row.names = 1)

obj <- Seurat::CreateSeuratObject(expr, meta.data = metadata)

centroids_data <- spatial[c('X', 'Y')]
rownames(centroids_data) <- NULL

centroids_data['cell'] <- rownames(obj@meta.data)
cents <- CreateCentroids(centroids_data)
coords <- CreateFOV(coords = list("centroids" = cents), type = "centroids")
obj[["fov"]]<-subset(coords, cell=Cells(obj))

## Prepare CellChat inputs

options(future.globals.maxSize = 8000 * 1024^2)
obj <- SCTransform(obj, assay = "RNA", clip.range = c(-10, 10), verbose = FALSE)

data.input = GetAssayData(obj, slot = "counts", assay = "RNA")

spatial.locs = as.matrix(GetTissueCoordinates(obj, scale = NULL)[c('x', 'y')])

meta = obj@meta.data[c('donor_id', 'cluster')]

conversion.factor = 1.
spot.size = 6.57*conversion.factor 
spatial.factors = data.frame(ratio = conversion.factor, tol = spot.size/2)

# spatial.factors = data.frame(ratio = 0.18, tol = 0.05)

cellchat <- createCellChat(object = data.input, meta = meta, group.by = 'cluster',
                           datatype = "spatial", coordinates = spatial.locs,
                           spatial.factors = spatial.factors)

CellChatDB <- CellChatDB.human

# CellChatDB.use <- subsetDB(CellChatDB, search = "Cell-Cell Contact", key = "annotation") # use Secreted Signaling

# use a subset of CellChatDB for cell-cell communication analysis
# CellChatDB.use <- subsetDB(CellChatDB, search = "Secreted Signaling", key = "annotation") # use Secreted Signaling

# Only uses the Secreted Signaling from CellChatDB v1
#  CellChatDB.use <- subsetDB(CellChatDB, search = list(c("Secreted Signaling"), c("CellChatDB v1")), key = c("annotation", "version"))

# use all CellChatDB except for "Non-protein Signaling" for cell-cell communication analysis
CellChatDB.use <- subsetDB(CellChatDB)

# use all CellChatDB for cell-cell communication analysis
# CellChatDB.use <- CellChatDB # simply use the default CellChatDB. We do not suggest to use it in this way because CellChatDB v2 includes "Non-protein Signaling" (i.e., metabolic and synaptic signaling) that can be only estimated from gene expression data.

cellchat@DB <- CellChatDB.use

# subset the expression data of signaling genes for saving computation cost
cellchat <- subsetData(cellchat) # This step is necessary even if using the whole database

cellchat <- identifyOverExpressedGenes(cellchat)
cellchat <- identifyOverExpressedInteractions(cellchat, variable.both = F)

cellchat <- computeCommunProb(cellchat, type = "truncatedMean", trim = 0.1,
                              distance.use = TRUE, interaction.range = 50, 
                              scale.distance = 1.,
                              contact.dependent = TRUE, contact.range = 10)


cellchat <- filterCommunication(cellchat, min.cells = 10)

cellchat <- computeCommunProbPathway(cellchat)

cellchat <- aggregateNet(cellchat)

# groupSize <- as.numeric(table(cellchat@idents))
# par(mfrow = c(1,2), xpd=TRUE)
# netVisual_circle(cellchat@net$count, vertex.weight = rowSums(cellchat@net$count), weight.scale = T, label.edge= F, title.name = "Number of interactions")
# netVisual_circle(cellchat@net$weight, vertex.weight = rowSums(cellchat@net$weight), weight.scale = T, label.edge= F, title.name = "Interaction weights/strength")

# pdf(paste0("all_protein/", sample, ".pdf"), height=2.5, width=3)
# print(netVisual_heatmap(cellchat, measure = "count", color.heatmap = "Blues"))
# print(netVisual_heatmap(cellchat, measure = "weight", color.heatmap = "Blues"))
# dev.off()

saveRDS(cellchat, paste0("./Steamboat_experiments/cellchat/all_protein/slidetags.rds"))
write.csv(cellchat@net$weight, paste0("./Steamboat_experiments/cellchat/all_protein/slidetags.csv"))

