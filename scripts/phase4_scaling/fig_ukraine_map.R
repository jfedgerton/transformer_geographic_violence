# fig_ukraine_map.R -- Ukraine forecast maps for the fine-tuning exercise (fig:ukraine_invasion + appendix maps).
# Shading = forecast probability of any violent event in the month shown, from the pretrained encoder fine-tuned on
# the chosen share of Europe's training data, averaged over 3 encoder seeds x 5 draws. Black outlines = cells with
# at least one violent event that month. Panels are labelled by TARGET month (origin month + 1).
#
# Input : outputs/ukraine_maps/ukraine_cells_f{share}{suffix}.csv   (export_ukraine_map.py)
#         data_raw/shapefiles/ne_10m_admin_0_ukr/ne_10m_admin_0_countries_ukr.shp  (Natural Earth, Ukraine POV)
# Output: outputs/ukraine_maps/fig_ukraine_f{share}{suffix}_{set}.png and .pdf  (never overwritten)
# Run   : Rscript scripts/phase4_scaling/fig_ukraine_map.R <cells_csv> [shapefile] [out_dir]
# Added 2026-10-20 (generalizes outputs/ukraine_maps/fig_ukraine_ft025.R to any share).
suppressPackageStartupMessages({library(ggplot2); library(sf); library(dplyr); library(readr)})
set.seed(123)
ROOT <- "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) >= 1)
IN  <- args[1]
SHP <- if (length(args) >= 2) args[2] else file.path(ROOT, "data_raw/shapefiles/ne_10m_admin_0_ukr/ne_10m_admin_0_countries_ukr.shp")
OUT <- if (length(args) >= 3) args[3] else dirname(IN)
stem <- sub("^ukraine_cells_", "fig_ukraine_", sub("\\.csv$", "", basename(IN)))
stopifnot(file.exists(IN), file.exists(SHP), dir.exists(OUT), grepl("^fig_ukraine_f", stem))

# ---- 1. data and checks ----
d <- read_csv(IN, show_col_types = FALSE, col_types = cols(year_month = "c", target_month = "c"))
stopifnot(nrow(d) == 13200, sum(d$y) == 4745, all(d$p_mean >= 0 & d$p_mean <= 1),
          !any(duplicated(d[, c("grid_id", "target_month")])))
cell_poly <- function(lon, lat) st_polygon(list(rbind(c(lon-.25, lat-.25), c(lon+.25, lat-.25), c(lon+.25, lat+.25),
                                                      c(lon-.25, lat+.25), c(lon-.25, lat-.25))))
cells <- d %>% distinct(grid_id, lon, lat)
stopifnot(nrow(cells) == 330)
cells_sf <- st_sf(grid_id = cells$grid_id,
                  geometry = st_sfc(mapply(cell_poly, cells$lon, cells$lat, SIMPLIFY = FALSE), crs = 4326))
sf_use_s2(FALSE)
W <- st_read(SHP, quiet = TRUE) %>% st_make_valid()
BB <- st_bbox(c(xmin = 21.5, xmax = 41, ymin = 42.8, ymax = 53), crs = st_crs(4326))
W <- suppressWarnings(st_crop(W, BB))
UKR <- W %>% filter(ADMIN == "Ukraine"); NBR <- W %>% filter(ADMIN != "Ukraine")
stopifnot(nrow(UKR) == 1)
d_sf <- cells_sf %>% inner_join(d, by = "grid_id")
stopifnot(nrow(d_sf) == 13200)

# ---- 2. panels (six target months each) ----
SETS <- list(
  "1_invasion"           = c("2022-02", "2022-03", "2022-04", "2022-05", "2022-06", "2022-07"),
  "2_counteroffensives"  = c("2022-08", "2022-09", "2022-10", "2022-11", "2022-12", "2023-01"),
  "3_southern_offensive" = c("2023-06", "2023-07", "2023-08", "2023-09", "2023-10", "2023-11"),
  "4_2024_2025"          = c("2024-08", "2024-10", "2024-12", "2025-02", "2025-04", "2025-05"))
for (nm in names(SETS)) {
  f_png <- file.path(OUT, paste0(stem, "_", nm, ".png")); f_pdf <- sub("\\.png$", ".pdf", f_png)
  stopifnot(!file.exists(f_png), !file.exists(f_pdf))
  s <- d_sf %>% filter(target_month %in% SETS[[nm]])
  stopifnot(length(unique(s$target_month)) == 6, nrow(s) == 6 * 330)
  s$panel <- factor(format(as.Date(paste0(s$target_month, "-01")), "%b %Y"),
                    levels = format(as.Date(paste0(SETS[[nm]], "-01")), "%b %Y"))
  p <- ggplot() +
    geom_sf(data = NBR, fill = "grey93", color = "grey70", linewidth = 0.25) +
    geom_sf(data = s, aes(fill = p_mean), color = NA) +
    geom_sf(data = filter(s, y == 1), fill = NA, color = "black", linewidth = 0.35) +
    geom_sf(data = UKR, fill = NA, color = "grey15", linewidth = 0.6) +
    scale_fill_gradient(low = "#FFF5EB", high = "#7F2704", limits = c(0, 1), name = "Forecast\nprobability") +
    facet_wrap(~ panel, ncol = 3) +
    coord_sf(xlim = c(21.5, 41), ylim = c(42.8, 53), expand = FALSE) +
    theme_minimal(base_size = 10) +
    theme(panel.grid = element_blank(), axis.text = element_blank(), axis.title = element_blank(),
          legend.position = "right", strip.text = element_text(face = "bold"))
  ggsave(f_png, p, width = 9, height = 5.2, dpi = 300); ggsave(f_pdf, p, width = 9, height = 5.2)
  cat(nm, "rows", nrow(s), "violent", sum(s$y), "->", f_png, "\n")
}
cat("[done]\n")
