library(monocle3)
library(Matrix)
library(dplyr)
library(ggplot2)
library(cowplot)

set.seed(42)

project_args <- commandArgs(trailingOnly = TRUE)
base_path <- normalizePath(if (length(project_args) > 0) project_args[[1]] else Sys.getenv("TRAJRISK_ROOT", unset = "."), mustWork = TRUE)
monocle_dir <- file.path(base_path, "monocle_input")
results_dir <- file.path(base_path, "results/figures")
output_dir  <- file.path(base_path, "Data/processed")

# ── 1. Load exported data ─────────────────────────────────────────────────────

expr_matrix <- readMM(file.path(monocle_dir, "expression_matrix.mtx"))

cell_barcodes <- read.csv(file.path(monocle_dir, "cell_barcodes.csv"),
                          header = FALSE, stringsAsFactors = FALSE)[[1]]
gene_names    <- read.csv(file.path(monocle_dir, "gene_names.csv"),
                          header = FALSE, stringsAsFactors = FALSE)[[1]]

expr_matrix <- t(expr_matrix)
rownames(expr_matrix) <- gene_names
colnames(expr_matrix) <- cell_barcodes

cat("Expression matrix:", nrow(expr_matrix), "genes ×", ncol(expr_matrix), "cells\n")

cell_meta <- read.csv(file.path(monocle_dir, "cell_metadata.csv"),
                      row.names = 1, stringsAsFactors = FALSE)
cell_meta$timepoint_hours <- as.numeric(cell_meta$timepoint_hours)
cell_meta <- cell_meta[colnames(expr_matrix), , drop = FALSE]

scvi_latent <- as.matrix(read.csv(file.path(monocle_dir, "scvi_latent.csv"),
                                  row.names = 1))
scvi_latent <- scvi_latent[colnames(expr_matrix), ]

cat("Metadata rows:", nrow(cell_meta), "\n")
cat("scVI dimensions:", ncol(scvi_latent), "\n")

# ── 2. Subset to Walesky cells ────────────────────────────────────────────────

walesky_cells <- rownames(cell_meta)[cell_meta$dataset == "Walesky"]
cat("\nWalesky cells:", length(walesky_cells), "\n")

expr_w  <- expr_matrix[, walesky_cells]
meta_w  <- cell_meta[walesky_cells, ]
scvi_w  <- scvi_latent[walesky_cells, ]

cat("Walesky timepoints and treatment:\n")
print(table(meta_w$timepoint_str, meta_w$treatment))

# ── 3. Create Walesky CellDataSet ─────────────────────────────────────────────

gene_meta <- data.frame(
  gene_short_name = rownames(expr_w),
  row.names       = rownames(expr_w)
)

cds_w <- new_cell_data_set(
  expression_data = expr_w,
  cell_metadata   = meta_w,
  gene_metadata   = gene_meta
)

cds_w <- cds_w[rowSums(exprs(cds_w)) > 0, ]
cat("\nCellDataSet:", ncol(cds_w), "cells,", nrow(cds_w), "genes\n")

# ── 4. Inject scVI latent space and compute UMAP ──────────────────────────────

cds_w <- preprocess_cds(cds_w, num_dim = 30, method = "PCA", verbose = FALSE)
reducedDims(cds_w)[["PCA"]] <- scvi_w

cds_w <- reduce_dimension(
  cds_w,
  reduction_method  = "UMAP",
  preprocess_method = "PCA",
  umap.min_dist     = 0.1,
  umap.n_neighbors  = 15,
  verbose           = FALSE
)

cat("UMAP computed from scVI latent space\n")

# ── 5. Cluster cells ──────────────────────────────────────────────────────────

cds_w <- cluster_cells(cds_w, reduction_method = "UMAP", resolution = 1e-3)

cat("Clusters:", length(unique(clusters(cds_w))),
    "| Partitions:", length(unique(partitions(cds_w))), "\n")

# ── 6. Learn trajectory graph ─────────────────────────────────────────────────

cds_w <- learn_graph(cds_w, use_partition = FALSE, verbose = FALSE)
cat("Trajectory graph learned\n")

# ── 7. Order cells by pseudotime ──────────────────────────────────────────────

timepoint_order <- c("0h", "6h", "24h", "48h", "96h")
colData(cds_w)$timepoint_str <- factor(colData(cds_w)$timepoint_str,
                                       levels = timepoint_order)

root_ids <- rownames(colData(cds_w))[
  colData(cds_w)$timepoint_str == "0h" &
    colData(cds_w)$treatment     == "Control"
]

cat("Root cells (0h Control):", length(root_ids), "\n")
cds_w <- order_cells(cds_w, root_cells = root_ids)
cat("Pseudotime calculated\n")

# ── 8. Validate pseudotime ────────────────────────────────────────────────────

pseudo_w <- pseudotime(cds_w)
tp_w     <- as.numeric(colData(cds_w)$timepoint_hours)
valid_w  <- is.finite(pseudo_w) & !is.na(tp_w)

rho_result <- cor.test(
  pseudo_w[valid_w], tp_w[valid_w],
  method = "spearman", exact = FALSE
)

cat("\n── TRAJECTORY VALIDATION ────────────────────────────────────\n")
cat("Valid cells:          ", sum(valid_w), "\n")
cat("Infinite pseudotime:  ", sum(!is.finite(pseudo_w)), "\n")
cat("Spearman ρ:           ", round(rho_result$estimate, 4), "\n")
cat("p-value:              ", format(rho_result$p.value, scientific = TRUE), "\n")

if (abs(rho_result$estimate) >= 0.50) {
  cat("PASS — moderate temporal ordering\n")
} else {
  cat("BORDERLINE — check trajectory\n")
}

cat("\nMedian pseudotime per timepoint:\n")
pt_summary <- aggregate(
  pseudo_w[valid_w],
  by  = list(timepoint = as.character(colData(cds_w)$timepoint_str[valid_w])),
  FUN = median, na.rm = TRUE
)
colnames(pt_summary)[2] <- "median_pseudotime"
pt_summary$timepoint <- factor(pt_summary$timepoint, levels = timepoint_order)
print(pt_summary[order(pt_summary$timepoint), ])
cat("────────────────────────────────────────────────────────────\n")

# ── 9. Visualize trajectory ───────────────────────────────────────────────────

p_tp <- plot_cells(cds_w, color_cells_by = "timepoint_str",
                   label_cell_groups = FALSE, label_leaves = FALSE,
                   label_branch_points = FALSE) +
  ggtitle("Walesky trajectory — timepoint")

p_pt <- plot_cells(cds_w, color_cells_by = "pseudotime",
                   label_cell_groups = FALSE, label_leaves = FALSE,
                   label_branch_points = FALSE) +
  scale_color_viridis_c(option = "plasma") +
  ggtitle("Walesky trajectory — pseudotime")

ggsave(
  file.path(results_dir, "monocle_walesky_trajectory.png"),
  plot_grid(p_tp, p_pt, ncol = 2),
  width = 14, height = 6, dpi = 300
)
cat("Trajectory figure saved\n")

# ── 10. Gene expression along pseudotime ──────────────────────────────────────

markers    <- c("Hmox1", "Gsta1", "Alb", "Cyp2e1")
markers_in <- markers[markers %in% rownames(cds_w)]

if (length(markers_in) > 0) {
  p_genes <- plot_genes_in_pseudotime(
    cds_w[markers_in, ],
    color_cells_by = "timepoint_str",
    min_expr       = 0.01,
    ncol           = 2
  )
  ggsave(
    file.path(results_dir, "monocle_walesky_genes_pseudotime.png"),
    p_genes, width = 12, height = 8, dpi = 300
  )
  cat("Gene expression along pseudotime saved\n")
}

# ── 11. Save ──────────────────────────────────────────────────────────────────

pseudo_df <- data.frame(
  cell_id    = names(pseudo_w),
  pseudotime = as.numeric(pseudo_w),
  row.names  = names(pseudo_w)
)

write.csv(pseudo_df, file.path(output_dir, "pseudotime_walesky.csv"), quote = FALSE)
saveRDS(cds_w, file.path(output_dir, "monocle_cds_walesky.rds"))

cat("\n✅ pseudotime_walesky.csv saved\n")
cat("✅ monocle_cds_walesky.rds saved\n")
cat("\nFinal summary:\n")
cat("  Cells:            ", ncol(cds_w), "\n")
cat("  Valid pseudotime: ", sum(valid_w), "\n")
cat("  Spearman ρ:       ", round(rho_result$estimate, 4), "\n")
cat("  Pseudotime range:  0 —",
    round(max(pseudo_w[is.finite(pseudo_w)]), 3), "\n")
