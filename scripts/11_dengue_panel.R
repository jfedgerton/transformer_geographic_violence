R_DIR <- "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
suppressMessages(library(data.table))

cent <- fread(file.path(R_DIR, "data_raw/br_centroids/municipios.csv"))
setnames(cent, "codigo_ibge", "MUNICIP")

cols_need <- c("MUNICIP","WEEK","WEEK_NO","YEAR","YEAR_NO","MONTH","CASES","pop")
csv_path <- file.path(R_DIR, "data_intermediate/dengue_brazil_raw.csv")
df <- fread(csv_path, select = cols_need)
cat("[load] rows:", nrow(df), "\n")

n_before_na <- nrow(df)
df <- df[!is.na(YEAR) & !is.na(WEEK_NO)]
cat("[clean] dropped", n_before_na - nrow(df), "rows with NA YEAR/WEEK_NO\n")

df <- merge(df, cent[, .(MUNICIP, latitude, longitude)], by = "MUNICIP", all.x = TRUE)
n_before <- nrow(df)
df <- df[!is.na(latitude)]
cat("[join] dropped", n_before - nrow(df), "rows with no centroid match\n")

cat("[check] YEAR range:", range(df$YEAR, na.rm = TRUE), " WEEK_NO range:", range(df$WEEK_NO, na.rm = TRUE), "\n")
df[, t := (YEAR - min(YEAR, na.rm =TRUE)) * 53L + WEEK_NO]
cat("[check] t range:", range(df$t), " n_unique_t:", length(unique(df$t)), "\n")

df[, inc := CASES / pmax(pop, 1) * 1000]
df[, log_inc := log1p(inc)]
df[, `:=`(mu = mean(log_inc), sd = sd(log_inc)), by = MUNICIP]
df[, sd := pmax(sd, 1e-6)]
df[, z := (log_inc - mu) / sd]

sev <- function(z) {
  fifelse(z >= 2.0, 4L, fifelse(z >= 1.5, 3L, fifelse(z >= 1.0, 2L, fifelse(z >= 0.5, 1L, 0L))))
}
df[, severity := sev(z)]
df[, pos := as.integer(severity > 0)]

cat("[severity] table:\n"); print(table(df$severity))
cat("[pos rate]", mean(df$pos), "\n")

muni_meta <- df[, .(lat = mean(latitude), lon = mean(longitude), pop = mean(pop)), by = MUNICIP]
cat("[meta] n municipalities:", nrow(muni_meta), "\n")

out <- df[, .(MUNICIP, t, YEAR, WEEK_NO, severity, pos)]
out <- merge(out, muni_meta, by = "MUNICIP", all.x = TRUE)
out <- out[!is.na(pos)]
out <- out[order(MUNICIP, t)]

OUT_PATH <- file.path(R_DIR, "data_intermediate/dengue_panel_raw.csv.gz")
fwrite(out, OUT_PATH, compress = "gzip")
cat("[done] wrote", OUT_PATH, "rows=", nrow(out), "\n")
