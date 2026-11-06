# fig_brazil_deaths_map.R (2026-11-06) -- map of total deaths per PRIO-GRID cell for the four Brazil SIM outcomes
# (Appendix app:vocab). Reads outputs/crossdomain/brazil_cell_totals.csv (written by brazil_descriptives.py) and the
# Natural Earth 1:50m country file; writes figures_tables/fig_brazil_deaths_map.{pdf,png}.
# Usage: Rscript scripts/crossdomain/fig_brazil_deaths_map.R
suppressPackageStartupMessages({
  library(sf); library(ggplot2); library(dplyr); library(readr); library(scales)
})
ROOT <- "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
IN_CSV <- file.path(ROOT, "outputs/crossdomain/brazil_cell_totals.csv")
NE_ZIP <- file.path(ROOT, "data_raw/shapefiles/ne_50m_admin_0/ne_50m_admin_0_countries.zip")
OUT_PDF <- file.path(ROOT, "figures_tables/fig_brazil_deaths_map.pdf")
OUT_PNG <- file.path(ROOT, "figures_tables/fig_brazil_deaths_map.png")
stopifnot(file.exists(IN_CSV), file.exists(NE_ZIP))

cells <- read_csv(IN_CSV, show_col_types = FALSE)
stopifnot(all(c("outcome", "grid_id", "lon", "lat", "deaths") %in% names(cells)))
stopifnot(setequal(unique(cells$outcome), c("Homicide", "Traffic", "Suicide", "Drowning")))
stopifnot(all(table(cells$outcome) == 1478), all(cells$deaths >= 0))
cells$outcome <- factor(cells$outcome, levels = c("Homicide", "Traffic", "Suicide", "Drowning"))
cat("[load] cells:", nrow(cells), " total deaths by outcome:\n")
print(tapply(cells$deaths, cells$outcome, sum))

world <- st_read(paste0("/vsizip/", NE_ZIP), quiet = TRUE)
name_col <- intersect(c("ADMIN", "NAME", "NAME_EN"), names(world))[1]
stopifnot(!is.na(name_col))
brazil <- world[world[[name_col]] == "Brazil", ]
stopifnot(nrow(brazil) == 1)

# zero-death cells are drawn in light gray; positive totals on one shared log scale across the four panels
cells_pos <- filter(cells, deaths > 0)
cells_zero <- filter(cells, deaths == 0)
cat("[cells] zero-total cells by outcome:\n"); print(table(cells_zero$outcome))

p <- ggplot() +
  geom_tile(data = cells_zero, aes(x = lon, y = lat), width = 0.5, height = 0.5, fill = "gray88") +
  geom_tile(data = cells_pos, aes(x = lon, y = lat, fill = deaths), width = 0.5, height = 0.5) +
  geom_sf(data = brazil, fill = NA, color = "gray25", linewidth = 0.25) +
  scale_fill_distiller(palette = "Reds", direction = 1, trans = "log10", labels = label_comma(),
                       name = "Total deaths (log scale)",
                       guide = guide_colorbar(title.position = "top", barwidth = unit(6, "cm"), barheight = unit(0.35, "cm"))) +
  facet_wrap(~ outcome, ncol = 2) +
  coord_sf(xlim = c(-74.5, -34), ylim = c(-34, 5.8), expand = FALSE) +
  theme_void(base_size = 11) +
  theme(strip.text = element_text(size = 11, margin = margin(b = 4)),
        legend.position = "bottom",
        legend.title = element_text(size = 9),
        legend.text = element_text(size = 8),
        panel.spacing = unit(0.8, "lines"),
        plot.background = element_rect(fill = "white", color = NA))

ggsave(OUT_PDF, p, width = 7.5, height = 7.5)
ggsave(OUT_PNG, p, width = 7.5, height = 7.5, dpi = 300)
cat("[write]", OUT_PDF, "\n[write]", OUT_PNG, "\nBRDESC_MAP_DONE\n")
