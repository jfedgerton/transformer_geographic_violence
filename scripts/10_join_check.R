R_DIR <- "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
library(data.table)

cent <- fread(file.path(R_DIR, "data_raw/br_centroids/municipios.csv"))
cat("centroid rows:", nrow(cent), "cols:", paste(colnames(cent), collapse=","), "\n")

dengue_csv <- file.path(R_DIR, "data_intermediate/dengue_brazil_raw.csv.gz")
# only need distinct MUNICIP codes for coverage check -- read just that column
cmd <- sprintf("zcat %s | cut -d',' -f1", dengue_csv)
muni_col <- fread(cmd = cmd, header = TRUE)
colnames(muni_col) <- "MUNICIP"
uniq_muni <- unique(muni_col$MUNICIP)
cat("n_unique MUNICIP in dengue data:", length(uniq_muni), "\n")

matched <- sum(uniq_muni %in% cent$codigo_ibge)
cat("matched to centroid table:", matched, "of", length(uniq_muni),
    sprintf("(%.2f%%)", 100*matched/length(uniq_muni)), "\n")

unmatched <- uniq_muni[!(uniq_muni %in% cent$codigo_ibge)]
cat("n unmatched:", length(unmatched), "\n")
if (length(unmatched) > 0) {
  cat("sample unmatched codes:", paste(head(unmatched, 10), collapse=","), "\n")
}
cat("[done]\n")
