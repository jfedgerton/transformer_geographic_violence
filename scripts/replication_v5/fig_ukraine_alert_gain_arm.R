# fig_ukraine_alert_gain_arm.R -- replication_v5 copy of scripts/phase4_scaling/fig_ukraine_alert_gain.R (2026-11-02).
# Only change: the fine-tuning arm in the three output names comes from env UKR_ARM (top2 | full, required) instead of
# the fixed "top2", so the full-arm Ukraine figure can be built beside the top-two one. Inputs and logic unchanged.
# fig_ukraine_alert_gain.R -- written 2026-10-29 (Claude, scoped request from Jared).
# Purpose: two Ukraine panels side by side showing that the pretrained encoder's advantage is large when
#   local fine-tuning data is scarce and gone when it is plentiful. Replaces the probability-shaded
#   fig_ukraine_share_grid.R, which could not show the claim because the claim lives in ranking, not level.
#   Left panel  = 0.25 percent of Europe's training data, right panel = 10 percent.
#   Within each panel, over the post window 2022-02..2025-05 (40 target months, 330 cells):
#     alert budget k = the number of violent cell-months in the window, so both models flag the same k
#     cell-months (the equal-budget convention of figures_tables/fig_cell_alerts.R, Figure 3);
#     per cell, net = (true alarms gained) - (false alarms added) by the pretrained model relative to the
#     from-scratch model, summed over the window.
#   Shading = net, diverging, blue where the pretrained model gains and red where it loses, on one
#     symmetric scale shared by both panels. Figure 3's palette and dark land are reused so the two figures
#     sit together. Cells with events but net exactly 0 take Figure 3's muted "No change" grey; cells with
#     neither events nor change are left as unshaded land.
# Inputs : <cells_csv_0.0025> <cells_csv_0.1>  from export_ukraine_map_fullsrc.py (SIZE=M, --ft top2)
#          data_raw/shapefiles/ne_10m_admin_0_ukr/ne_10m_admin_0_countries_ukr.shp (Natural Earth, Ukraine POV)
# Outputs: <out_dir>/fig_ukraine_alert_gain_M_fullsrc_top2.pdf and .png (300 dpi). Never overwritten.
#          <out_dir>/ukraine_alert_gain_cells_M_fullsrc_top2.csv (the per-cell numbers behind the figure).
# Run    : Rscript scripts/phase4_scaling/fig_ukraine_alert_gain.R <csv_0.0025> <csv_0.1> <out_dir> [shapefile]
#          Run it as a Slurm job: the login node has no libudunits2, so sf will not load there.
#          See scripts/phase4_scaling/launch_ukraine_alert_gain.sbatch.
suppressPackageStartupMessages({library(ggplot2); library(sf); library(dplyr); library(readr); library(scales)})
set.seed(123)
ARM <- Sys.getenv("UKR_ARM"); stopifnot(ARM %in% c("top2", "full"))
ROOT <- "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) >= 3)
IN_LO <- args[1]; IN_HI <- args[2]; OUT <- args[3]
SHP <- if (length(args) >= 4) args[4] else file.path(ROOT, "data_raw/shapefiles/ne_10m_admin_0_ukr/ne_10m_admin_0_countries_ukr.shp")
stopifnot(file.exists(IN_LO), file.exists(IN_HI), file.exists(SHP), dir.exists(OUT))
MONTHS <- format(seq(as.Date("2022-02-01"), as.Date("2025-05-01"), by = "month"), "%Y-%m")
stopifnot(length(MONTHS) == 40)

# ---- 1. equal alert budget, then per-cell true alarms gained minus false alarms added ----
# Budget convention of figures_tables/fig_cell_alerts.R: k = violent cell-months in the window, so the two
# models flag the same number of cell-months. Ties are broken by the stable order of the exported rows,
# which is the same rule the alert code in export_ukraine_map_fullsrc.py uses.
flag_top_k <- function(p, k) { f <- integer(length(p)); f[order(-p)[seq_len(k)]] <- 1L; f }
read_share <- function(path, lab) {
  d <- read_csv(path, show_col_types = FALSE, col_types = cols(year_month = "c", target_month = "c"))
  stopifnot(nrow(d) == 13200, sum(d$y) == 4745,
            all(d$p_mean >= 0 & d$p_mean <= 1), all(d$p_scratch >= 0 & d$p_scratch <= 1),
            identical(sort(unique(d$target_month)), MONTHS),
            !any(duplicated(d[, c("grid_id", "target_month")])))
  k <- sum(d$y)
  f_pre <- flag_top_k(d$p_mean, k); f_scr <- flag_top_k(d$p_scratch, k)
  stopifnot(sum(f_pre) == k, sum(f_scr) == k)
  d$d_tp <- f_pre * d$y - f_scr * d$y
  d$d_fp <- f_pre * (1 - d$y) - f_scr * (1 - d$y)
  g <- d %>% group_by(grid_id, lon, lat) %>%
    summarise(n_month = n(), events = sum(y), d_tp = sum(d_tp), d_fp = sum(d_fp), .groups = "drop") %>%
    mutate(net = d_tp - d_fp, share_lab = lab)
  stopifnot(nrow(g) == 330, all(g$n_month == 40), sum(g$d_tp) + sum(g$d_fp) == 0)
  cat(sprintf("[%s] budget k = %d, TP pretrained %d, TP scratch %d, net sum %d\n",
              lab, k, sum(f_pre * d$y), sum(f_scr * d$y), sum(g$net)))
  g
}
LAB_LO <- "0.25% of Europe's training data"
LAB_HI <- "10% of Europe's training data"
cells <- bind_rows(read_share(IN_LO, LAB_LO), read_share(IN_HI, LAB_HI)) %>%
  mutate(share_lab = factor(share_lab, levels = c(LAB_LO, LAB_HI)))
stopifnot(nrow(cells) == 660, !any(is.na(cells$net)), !any(is.na(cells$share_lab)))

# ---- 2. numeric summary, so the figure can be checked before anyone looks at it ----
summ <- cells %>% group_by(share_lab) %>%
  summarise(cells_gain = sum(net > 0), cells_loss = sum(net < 0), cells_zero = sum(net == 0),
            zero_with_events = sum(net == 0 & events > 0), zero_no_events = sum(net == 0 & events == 0),
            net_sum = sum(net), net_mean = round(mean(net), 3),
            net_min = min(net), net_max = max(net), .groups = "drop")
print(as.data.frame(summ))
f_csv <- file.path(OUT, paste0("ukraine_alert_gain_cells_M_fullsrc_", ARM, ".csv"))
stopifnot(!file.exists(f_csv))
write_csv(cells, f_csv)

# ---- 3. geometry (same shapefile, projection and extent as fig_ukraine_map.R) ----
sf_use_s2(FALSE)
W <- st_read(SHP, quiet = TRUE) %>% st_make_valid()
BB <- st_bbox(c(xmin = 21.5, xmax = 41, ymin = 42.8, ymax = 53), crs = st_crs(4326))
W <- suppressWarnings(st_crop(W, BB))
UKR <- W %>% filter(ADMIN == "Ukraine"); NBR <- W %>% filter(ADMIN != "Ukraine")
stopifnot(nrow(UKR) == 1)

# ---- 4. figure (palette and dark land from figures_tables/fig_cell_alerts.R) ----
LAND <- "#3A3A3A"; BORDER <- "#4A4A4A"; OUTLINE <- "#D0D0D0"; NOCHANGE <- "#5E5E5E"
LIM <- 20   # symmetric, shared by both panels; |net| beyond this is squished, as Figure 3 does
cat("cells squished at |net| >", LIM, ":", sum(abs(cells$net) > LIM), "of", nrow(cells), "\n")
chg <- cells %>% filter(net != 0)
nochg <- cells %>% filter(net == 0, events > 0)
cat("drawn: changed", nrow(chg), " grey no-change-with-events", nrow(nochg),
    " left as land", nrow(cells) - nrow(chg) - nrow(nochg), "\n")
leg_title <- "Correct alerts gained minus false alerts added by the pretrained model, per 0.5-degree cell\n(count of cell-months, 2022-02 to 2025-05, equal alert budgets)"
f_pdf <- file.path(OUT, paste0("fig_ukraine_alert_gain_M_fullsrc_", ARM, ".pdf"))
f_png <- file.path(OUT, paste0("fig_ukraine_alert_gain_M_fullsrc_", ARM, ".png"))
stopifnot(!file.exists(f_pdf), !file.exists(f_png))
p <- ggplot() +
  geom_sf(data = NBR, fill = LAND, colour = BORDER, linewidth = 0.15) +
  geom_sf(data = UKR, fill = LAND, colour = BORDER, linewidth = 0.15) +
  geom_tile(data = nochg, aes(lon, lat, colour = "Events, no change"), fill = NOCHANGE,
            width = 0.5, height = 0.5, linewidth = 0) +
  geom_tile(data = chg, aes(lon, lat, fill = net), width = 0.5, height = 0.5) +
  geom_sf(data = UKR, fill = NA, colour = OUTLINE, linewidth = 0.4) +
  scale_fill_gradientn(colours = c("#FF3B30", "#C8463F", "#6E6E6E", "#3F7BC8", "#40A6FF"),
                       values = rescale(c(-LIM, -LIM / 3, 0, LIM / 3, LIM)),
                       limits = c(-LIM, LIM), oob = squish, breaks = seq(-LIM, LIM, by = 10),
                       labels = c("<= -20", "-10", "0", "+10", ">= +20"), name = leg_title) +
  scale_colour_manual(values = c("Events, no change" = NOCHANGE), name = NULL) +
  facet_wrap(~ share_lab, ncol = 2) +
  coord_sf(xlim = c(21.5, 41), ylim = c(42.8, 53), expand = FALSE, datum = NA) +
  guides(fill = guide_colourbar(title.position = "top", title.hjust = 0.5, order = 1,
                                barwidth = unit(16, "lines"), barheight = unit(0.6, "lines")),
         colour = guide_legend(order = 2, override.aes = list(fill = NOCHANGE, linewidth = 0))) +
  labs(x = NULL, y = NULL) +
  theme_void(base_size = 10) +
  theme(legend.position = "bottom", legend.box = "horizontal", legend.title = element_text(size = 9),
        strip.text = element_text(face = "bold", size = 11, margin = margin(b = 5)),
        panel.spacing = unit(0.9, "lines"), plot.margin = margin(6, 6, 6, 6))
ggsave(f_pdf, p, width = 9.0, height = 5.0)
ggsave(f_png, p, width = 9.0, height = 5.0, dpi = 300)
cat("[done]", f_pdf, "\n[done]", f_png, "\n[done]", f_csv, "\n")
