library(Seurat)
library(CellChat)

future::plan("multisession", workers = 8)

full_obj <- readRDS("../data/ST_Discovery_so.rds")
sample_meta <- readxl::read_excel("../data/sample_metadata.xlsx", sheet = "Table 2b", skip = 1)

mask = (sample_meta['sites_binary'] == 'Adnexa') & (sample_meta['treatment'] == 'Untreated')
samples_of_interest = sample_meta$profile[mask]

sample_meta
samples_of_interest

table(full_obj$samples)
full_obj <- full_obj[, full_obj$samples %in% samples_of_interest]

for (sample in unique(full_obj$samples)[10:length(unique(full_obj$samples))]){

    obj <- full_obj[, full_obj$samples == sample]
    
    centroids_data <- obj@meta.data[c('x', 'y')]
    rownames(centroids_data) <- NULL
    centroids_data['cell'] <- rownames(obj@meta.data)
    cents <- CreateCentroids(centroids_data)
    coords <- CreateFOV(coords = list("centroids" = cents), type = "centroids")
    obj[["fov"]]<-subset(coords, cell=Cells(obj))

    ## Prepare CellChat inputs

    options(future.globals.maxSize = 8000 * 1024^2)
    obj <- SCTransform(obj, assay = "RNA", clip.range = c(-10, 10), verbose = FALSE)

    data.input = GetAssayData(obj, slot = "data", assay = "SCT")

    spatial.locs = as.matrix(GetTissueCoordinates(obj, scale = NULL)[c('x', 'y')])

    nolc <- function(x){
      if (stringr::str_sub(x, -3, -1) == '_LC') {
        return(stringr::str_sub(x, 0, -4))
      } else {
        return(x)
      }
    }
    obj$cell.types.nolc = sapply(obj$cell.types, nolc)
    table(obj$cell.types.nolc)

    meta = obj@meta.data[c('samples', 'cell.types.nolc')]

    spatial.factors = data.frame(ratio = 0.18, tol = 0.05)

    cellchat <- createCellChat(object = data.input, meta = meta, group.by = 'cell.types.nolc',
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
                                  distance.use = TRUE, interaction.range = 100, scale.distance = 6.2,
                                  contact.dependent = TRUE, contact.range = 20)


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
    
    saveRDS(cellchat, paste0("all_protein/", sample, ".rds"))
    write.csv(cellchat@net$weight, paste0("all_protein/", sample, ".csv"))
    
    rm(obj)
    gc()
}
