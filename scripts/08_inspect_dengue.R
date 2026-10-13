# 08_inspect_dengue.R
# Inspect the Brazil dengue RDS (residence-based) and convert to parquet for
# the Python pipeline. Run sequentially, fail loudly.

R_DIR <- "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
IN_PATH <- file.path(R_DIR, "data_raw/dengue_brazil/denv_resid.RDS")
OUT_PATH <- file.path(R_DIR, "data_intermediate/dengue_brazil_raw.parquet")

stopifnot(file.exists(IN_PATH))
df <- readRDS(IN_PATH)

cat("class:", class(df), "\n")
cat("dim:", dim(df), "\n")
cat("colnames:\n")
print(colnames(df))
cat("\nstr (first columns):\n")
str(df[1:min(10, ncol(df))], list.len = 20)
cat("\nhead:\n")
print(head(df, 3))

# look specifically for the fields we need to build the panel
cat("\n-- key field checks --\n")
for (cand in c("mun_code", "muni_code", "mun", "ibge_code", "code_muni",
               "week", "epiweek", "date", "year", "DENV", "denv", "cases",
               "lat", "lon", "latitude", "longitude")) {
  if (cand %in% colnames(df)) {
    cat(sprintf("FOUND '%s' -- class=%s n_unique=%d\n", cand, class(df[[cand]]),
                length(unique(df[[cand]]))))
  }
}

if (requireNamespace("arrow", quietly = TRUE)) {
  dir.create(dirname(OUT_PATH), showWarnings = FALSE, recursive = TRUE)
  arrow::write_parquet(df, OUT_PATH)
  cat("\nWrote parquet:", OUT_PATH, "\n")
} else {
  cat("\narrow package not available -- writing CSV fallback instead\n")
  out_csv <- sub("\\.parquet$", ".csv.gz", OUT_PATH)
  dir.create(dirname(out_csv), showWarnings = FALSE, recursive = TRUE)
  write.csv(df, gzfile(out_csv), row.names = FALSE)
  cat("Wrote csv.gz:", out_csv, "\n")
}
cat("[done]\n")
