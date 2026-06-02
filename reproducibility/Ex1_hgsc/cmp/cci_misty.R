# MISTy
library(mistyR)
library(future)

# data manipulation
library(dplyr)
library(purrr)
library(distances)

# plotting
library(ggplot2)

plan(multisession)

library(Seurat)
library(CellChat)

nolc <- function(x){
  if (stringr::str_sub(x, -3, -1) == '_LC') {
    return(stringr::str_sub(x, 0, -4))
  } else {
    return(x)
  }
}

full_obj <- readRDS("G:/data/HGSC/SeuratObj/ST_Discovery_so.rds")
sample_meta <- readxl::read_excel("G:/data/HGSC/sample_metadata.xlsx", sheet = "Table 2b", skip = 1, )

mask = (sample_meta['sites_binary'] == 'Adnexa') & (sample_meta['treatment'] == 'Untreated')
samples_of_interest = sample_meta$profile[mask]

sample_meta
samples_of_interest

table(full_obj$samples)
full_obj <- full_obj[, full_obj$samples %in% samples_of_interest]

for (sample in unique(full_obj$samples)){
  break
}

obj <- full_obj[, full_obj$samples == sample]
centroids_data <- obj@meta.data[c('x', 'y')]
rownames(centroids_data) <- NULL
centroids_data['cell'] <- rownames(obj@meta.data)
cents <- CreateCentroids(centroids_data)
coords <- CreateFOV(coords = list("centroids" = cents), type = "centroids")
obj[["fov"]]<-subset(coords, cell=Cells(obj))

# Expression data
expr <- as_tibble(t(GetAssayData(
  object = obj,
  slot = "counts",
  assay = "RNA"
)))

expr <- janitor::clean_names(expr)

# Seurat deals with duplicates internally in similar way as above

# Location data
geometry <- as_tibble(GetTissueCoordinates(obj, scale = NULL)[, c('x', 'y')])
colnames(geometry) <- c('col', 'row')

misty.intra <- create_initial_view(expr)
misty.views <- misty.intra %>% add_paraview(geometry, l=200)

run_misty(misty.views, paste0("misty_hgsc_", sample))


for (sample in unique(full_obj$samples)){
  gc()
  obj <- full_obj[, full_obj$samples == sample]
  centroids_data <- obj@meta.data[c('x', 'y')]
  rownames(centroids_data) <- NULL
  centroids_data['cell'] <- rownames(obj@meta.data)
  cents <- CreateCentroids(centroids_data)
  coords <- CreateFOV(coords = list("centroids" = cents), type = "centroids")
  obj[["fov"]] <- subset(coords, cell=Cells(obj))
  
  
  obj$cell.types.nolc = sapply(obj$cell.types, nolc)
  cell_types_factor <- factor(obj$cell.types.nolc)
  one_hot_matrix <- model.matrix(~ cell_types_factor - 1)
  colnames(one_hot_matrix) <- levels(cell_types_factor)
  expr <- as_tibble(one_hot_matrix)
  
  
  # expr <- as_tibble(t(GetAssayData(
  #   object = obj,
  #   slot = "counts",
  #   assay = "RNA"
  # )))
  # expr <- janitor::clean_names(expr)
  
  # Seurat deals with duplicates internally in similar way as above
  
  # Location data
  geometry <- as_tibble(GetTissueCoordinates(obj, scale = NULL)[, c('x', 'y')])
  colnames(geometry) <- c('col', 'row')
  
  misty.intra <- create_initial_view(expr)
  misty.views <- misty.intra %>% add_paraview(geometry, l=200)
  
  run_misty(misty.views, paste0("G:/Projects/Steamboat_experiments/misty/misty_hgsc_", sample))
}

# misty.results <- collect_results("G:/Projects/misty_hgsc_SMI_T10_F001/")
# saveRDS(misty.results, "misty.RDS")

for (sample in unique(full_obj$samples)){
  misty.results <- collect_results(paste0("G:/Projects/Steamboat_experiments/misty/misty_hgsc_", sample))
  
  get_interaction_heatmap <- function (misty.results, view, cutoff = 1, trim = -Inf, 
                                       trim.measure = c("gain.R2", "multi.R2", "intra.R2", "gain.RMSE", "multi.RMSE", "intra.RMSE"), 
                                       clean = FALSE){
    trim.measure.type <- match.arg(trim.measure)
    assertthat::assert_that(("importances.aggregated" %in% names(misty.results)), 
                            msg = "The provided result list is malformed. Consider using collect_results().")
    assertthat::assert_that(("improvements.stats" %in% names(misty.results)), 
                            msg = "The provided result list is malformed. Consider using collect_results().")
    assertthat::assert_that((view %in% (misty.results$importances.aggregated %>% 
                                          dplyr::pull(view))), msg = "The selected view cannot be found in the results table.")
    inv <- sign((stringr::str_detect(trim.measure.type, "gain") | 
                   stringr::str_detect(trim.measure.type, "RMSE", negate = TRUE)) - 
                  0.5)
    targets <- misty.results$improvements.stats %>% dplyr::filter(measure == 
                                                                    trim.measure.type, inv * mean >= inv * trim) %>% dplyr::pull(target)
    plot.data <- misty.results$importances.aggregated %>% dplyr::filter(view == 
                                                                          !!view, Target %in% targets)
    plot.data
  }
  
  plot.data <- misty.results %>% get_interaction_heatmap("para.200", clean = TRUE) 
  write.csv(plot.data, file = paste0("G:/Projects/Steamboat_experiments/misty/misty_cci_hgsc_", sample, ".csv"))
}
