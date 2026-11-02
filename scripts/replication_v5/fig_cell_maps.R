# fig_cell_maps.R -- main-text Figure 4 (fig:cell_maps): four 0.5-degree cell maps of alerts changed by the embedding.
#   A = base + embedding vs. persistence, global     B = same, Africa
#   C = base + embedding vs. base (nested), global   D = same, Africa
# Each cell: correct alerts gained minus false alerts added (net = d_tp - d_fp), already averaged over the three encoder
# seeds in the input files (cell_alerts_agg.py output; one row per PRIO-GRID cell). Conventions (palette, bins, cell
# geometry, Natural Earth borders, held-out region outlines, map extents) copied from figures_tables/fig_cell_alerts.R.
# Reproduces figures/fig_cell_alerts_maps_pers_nested_v2.png (original script fig_maps_2row.R is lost).
#
# Note on totals: at equal alert budgets every extra correct alert displaces one false alert, so sum(d_fp) = -sum(d_tp)
# and sum(net) = 2 * sum(d_tp). The numbers in Sec. 4.1 (4,762 / 1,695 / 710 / 197) are the extra correct alerts,
# sum(d_tp) = sum(net) / 2; the script checks both.
#
# Usage: Rscript fig_cell_maps.R <persistence_csv_gz> <incidence_csv_gz> <ne_zip> <out_dir>
# Tested with R 4.3.3, ggplot2 3.4.4, sf 1.0.15.
set.seed(123)
for (p in c("tidyverse", "sf")) if (!requireNamespace(p, quietly = TRUE)) stop("R package ", p, " is not installed")
suppressPackageStartupMessages({ library(tidyverse); library(sf) })
sf_use_s2(FALSE)
# UTF-8 locale so the degree sign renders in the legend title (a C locale prints it as "..")
if (!l10n_info()$`UTF-8`) invisible(suppressWarnings(Sys.setlocale("LC_CTYPE", "C.UTF-8")))
stopifnot(l10n_info()$`UTF-8`)

# ---- arguments and outputs --------------------------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) stop("usage: Rscript fig_cell_maps.R <persistence_csv_gz> <incidence_csv_gz> <ne_zip> <out_dir>")
f_pers <- args[1]; f_inc <- args[2]; f_ne <- args[3]; out_dir <- args[4]
for (f in c(f_pers, f_inc, f_ne)) if (!file.exists(f)) stop("input not found: ", f)
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
f_png <- file.path(out_dir, "fig_cell_alerts_maps_pers_nested_M_fullsrc.png")
f_pdf <- file.path(out_dir, "fig_cell_alerts_maps_pers_nested_M_fullsrc.pdf")
for (f in c(f_png, f_pdf)) if (file.exists(f)) stop("refusing to overwrite existing output: ", f)

# ---- cells ------------------------------------------------------------------------------------------------
bin_levels <- c("-2 or less", "-2 to -0.5", "-0.5 to 0", "No change", "0 to +0.5", "+0.5 to +2", "+2 or more")
pers <- read_csv(f_pers, show_col_types = FALSE)
inc <- read_csv(f_inc, show_col_types = FALSE)
need_pers <- c("gid", "region", "events", "tp_pers", "fp_pers", "tp_emb", "fp_emb", "d_tp", "d_fp", "lat", "lon")
need_inc <- c("gid", "region", "events", "tp_base", "fp_base", "tp_emb", "fp_emb", "n_seeds", "d_tp", "d_fp", "lat", "lon")
if (!all(need_pers %in% names(pers))) stop("persistence file missing columns: ", paste(setdiff(need_pers, names(pers)), collapse = ", "))
if (!all(need_inc %in% names(inc))) stop("incidence file missing columns: ", paste(setdiff(need_inc, names(inc)), collapse = ", "))
stopifnot(!anyDuplicated(pers$gid), !anyDuplicated(inc$gid), nrow(pers) == nrow(inc), setequal(pers$gid, inc$gid))
stopifnot(!anyNA(pers[need_pers]), !anyNA(inc[need_inc]), all(inc$n_seeds == 3))
stopifnot(all(abs(pers$lat * 4 - round(pers$lat * 4)) < 1e-9), all(abs(pers$lon * 4 - round(pers$lon * 4)) < 1e-9))  # 0.5-deg centres
# d_tp / d_fp are the seed means of the per-seed differences: check against the stored seed-mean tp/fp columns
stopifnot(max(abs(pers$d_tp - (pers$tp_emb - pers$tp_pers))) < 1e-3, max(abs(pers$d_fp - (pers$fp_emb - pers$fp_pers))) < 1e-3)
stopifnot(max(abs(inc$d_tp - (inc$tp_emb - inc$tp_base))) < 1e-3, max(abs(inc$d_fp - (inc$fp_emb - inc$fp_base))) < 1e-3)

bin_net <- function(d) {
  d |> mutate(net = d_tp - d_fp,
              bin = factor(case_when(net <= -2 ~ "-2 or less", net <= -0.5 ~ "-2 to -0.5", net < 0 ~ "-0.5 to 0",
                                     net == 0 ~ "No change", net < 0.5 ~ "0 to +0.5", net < 2 ~ "+0.5 to +2",
                                     TRUE ~ "+2 or more"), levels = bin_levels))
}
pers <- bin_net(pers); inc <- bin_net(inc)
stopifnot(!anyNA(pers$bin), !anyNA(inc$bin))

# ---- totals check against the text (Sec. 4.1) -------------------------------------------------------------
totals <- tibble(
  panel = c("A", "B", "C", "D"),
  target = c(4762, 1695, 710, 197),
  sum_d_tp = c(sum(pers$d_tp), sum(pers$d_tp[pers$region == "africa"]), sum(inc$d_tp), sum(inc$d_tp[inc$region == "africa"])),
  sum_d_fp = c(sum(pers$d_fp), sum(pers$d_fp[pers$region == "africa"]), sum(inc$d_fp), sum(inc$d_fp[inc$region == "africa"])),
  sum_net = c(sum(pers$net), sum(pers$net[pers$region == "africa"]), sum(inc$net), sum(inc$net[inc$region == "africa"])),
  # Africa by map extent instead of held-out region label (for the error report only)
  sum_d_tp_extent = c(NA, sum(pers$d_tp[between(pers$lon, -25, 60) & between(pers$lat, -36, 38)]), NA,
                      sum(inc$d_tp[between(inc$lon, -25, 60) & between(inc$lat, -36, 38)]))
) |> mutate(half_net = sum_net / 2)
print(totals)
bad <- totals |> filter(abs(half_net - target) > 1 | abs(sum_d_tp - target) > 1)
if (nrow(bad) > 0) {
  print(bad)
  if (Sys.getenv("FIG_ALLOW_TOTAL_DIFF") == "1") {
    warning("totals differ from Sec. 4.1 (FIG_ALLOW_TOTAL_DIFF=1 set, drawing anyway)")
  } else stop("totals do not reproduce Sec. 4.1 within 1 for panel(s) ", paste(bad$panel, collapse = ", "),
       ". Columns: sum_d_tp = extra correct alerts (region label), half_net = sum(d_tp - d_fp)/2, sum_net = sum(d_tp - d_fp), ",
       "sum_d_tp_extent = Africa by map extent", ". Set FIG_ALLOW_TOTAL_DIFF=1 to draw anyway.")
} else cat("[fig] totals reproduce Sec. 4.1 (extra correct alerts = sum(net)/2): A 4,762, B 1,695, C 710, D 197\n")

# ---- countries and held-out regions (hard-coded from scripts/phase4_scaling/regions.py, as in fig_cell_alerts.R) --
region_of <- list(
  africa = c("Angola","Benin","Botswana","Burkina Faso","Burundi","Cameroon","Cape Verde","Central African Republic","Chad",
             "Comoros","Congo","Democratic Republic of Congo","Djibouti","Equatorial Guinea","Eritrea","Ethiopia","Gabon",
             "Gambia","Ghana","Guinea","Guinea-Bissau","Ivory Coast","Kenya","Lesotho","Liberia","Madagascar","Malawi","Mali",
             "Mauritania","Mauritius","Mozambique","Namibia","Niger","Nigeria","Republic of Congo","Rwanda",
             "Sao Tome and Principe","Senegal","Seychelles","Sierra Leone","Somalia","South Africa","South Sudan","Sudan",
             "Tanzania","Togo","Uganda","Zambia","Zimbabwe","eSwatini"),
  americas = c("Argentina","Bahamas","Belize","Bolivia","Brazil","Canada","Chile","Colombia","Costa Rica","Cuba",
               "Dominican Republic","Ecuador","El Salvador","Falkland Islands","French Guiana","Guatemala","Guyana","Haiti",
               "Honduras","Jamaica","Mexico","Nicaragua","Panama","Paraguay","Peru","Puerto Rico","Suriname",
               "Trinidad and Tobago","United States of America","Uruguay","Venezuela"),
  asia_pacific = c("Australia","Brunei","Cambodia","China","East Timor","Fiji","Indonesia","Japan","Laos","Malaysia","Mongolia",
                   "Myanmar","New Caledonia","New Zealand","North Korea","Papua New Guinea","Philippines","Solomon Islands",
                   "South Korea","Taiwan","Thailand","Timor-Leste","Vanuatu","Vietnam"),
  europe = c("Albania","Armenia","Austria","Azerbaijan","Belarus","Belgium","Bosnia-Herzegovina","Bulgaria","Croatia","Cyprus",
             "Czech Republic","Denmark","Estonia","Finland","France","Georgia","Germany","Greece","Greenland","Hungary",
             "Iceland","Ireland","Italy","Latvia","Lithuania","Luxembourg","Moldova","Montenegro","Netherlands",
             "North Macedonia","Norway","Poland","Portugal","Romania","Russia","Serbia","Slovakia","Slovenia","Spain",
             "Sweden","Switzerland","Ukraine","United Kingdom"),
  mena = c("Algeria","Bahrain","Egypt","Iran","Iraq","Israel","Jordan","Kuwait","Lebanon","Libya","Morocco","Oman","Palestine",
           "Qatar","Saudi Arabia","Syria","Tunisia","Turkey","United Arab Emirates","Western Sahara","Yemen"),
  south_asia = c("Afghanistan","Bangladesh","Bhutan","India","Kazakhstan","Kyrgyzstan","Nepal","Pakistan","Sri Lanka",
                 "Tajikistan","Turkmenistan","Uzbekistan"))
lookup <- tibble(proj_name = unlist(region_of), region_key = rep(names(region_of), lengths(region_of)))
alias <- c("Dem. Rep. Congo" = "Democratic Republic of Congo", "Congo" = "Republic of Congo",
           "Central African Rep." = "Central African Republic", "Côte d'Ivoire" = "Ivory Coast", "Eq. Guinea" = "Equatorial Guinea",
           "S. Sudan" = "South Sudan", "W. Sahara" = "Western Sahara", "Bosnia and Herz." = "Bosnia-Herzegovina",
           "Czechia" = "Czech Republic", "Macedonia" = "North Macedonia", "Dominican Rep." = "Dominican Republic",
           "Falkland Is." = "Falkland Islands", "Solomon Is." = "Solomon Islands", "N. Cyprus" = "Cyprus",
           "Somaliland" = "Somalia", "Kosovo" = "Serbia", "Hong Kong" = "China", "Macao" = "China", "Cabo Verde" = "Cape Verde")
tmp <- tempfile(); unzip(f_ne, exdir = tmp)
shp <- list.files(tmp, pattern = "\\.shp$", full.names = TRUE, recursive = TRUE)
stopifnot(length(shp) == 1)
world <- st_read(shp, quiet = TRUE) |>
  select(NAME) |> filter(NAME != "Antarctica") |>
  mutate(proj_name = ifelse(NAME %in% names(alias), alias[NAME], NAME)) |>
  left_join(lookup, by = "proj_name")
st_crs(world) <- 4326
stopifnot(sum(!is.na(world$region_key)) >= 150)
regions <- world |> filter(!is.na(region_key)) |> st_make_valid() |> group_by(region_key) |> summarise(.groups = "drop")
stopifnot(nrow(regions) == 6)

# ---- panels -----------------------------------------------------------------------------------------------
LAND <- "#3A3A3A"; BORDER <- "#4A4A4A"; REGION_LINE <- "#D0D0D0"
leg_title <- "Correct alerts gained minus false alerts added\nper 0.5° cell (count of cell-months, test period)"
fill_bins <- scale_fill_manual(values = setNames(c("#FF3B30", "#E8746C", "#B9908C", "#5E5E5E", "#8FA9C9", "#5E9BE6", "#40A6FF"),
                                                 bin_levels), drop = FALSE, name = leg_title)
AFR_X <- c(-25, 60); AFR_Y <- c(-36, 38)
GLB_X <- c(-130, 180); GLB_Y <- c(-50, 75)
# binned version: every cell with events or any change (as in fig_cell_alerts.R)
act_pers <- pers |> filter(events > 0 | net != 0)
act_inc <- inc |> filter(events > 0 | net != 0)
cat(sprintf("[fig] cells drawn: persistence %d, nested %d (of %d)\n", nrow(act_pers), nrow(act_inc), nrow(pers)))

pA <- ggplot() +
  geom_sf(data = world, fill = LAND, colour = BORDER, linewidth = 0.08) +
  geom_tile(data = act_pers, aes(lon, lat, fill = bin), width = 0.5, height = 0.5) +
  geom_sf(data = regions, fill = NA, colour = REGION_LINE, linewidth = 0.25) +
  fill_bins + coord_sf(xlim = GLB_X, ylim = GLB_Y, expand = FALSE, datum = NA) + labs(x = NULL, y = NULL) +
  theme_void(base_size = 12) + theme(legend.position = "none", plot.margin = margin(0, 0, 0, 0))
pB <- ggplot() +
  geom_sf(data = world, fill = LAND, colour = BORDER, linewidth = 0.08) +
  geom_tile(data = act_pers, aes(lon, lat, fill = bin), width = 0.5, height = 0.5) +
  geom_sf(data = regions, fill = NA, colour = REGION_LINE, linewidth = 0.25) +
  fill_bins + coord_sf(xlim = AFR_X, ylim = AFR_Y, expand = FALSE, datum = NA) + labs(x = NULL, y = NULL) +
  theme_void(base_size = 12) + theme(legend.position = "none", plot.margin = margin(0, 0, 0, 0))
pC <- ggplot() +
  geom_sf(data = world, fill = LAND, colour = BORDER, linewidth = 0.08) +
  geom_tile(data = act_inc, aes(lon, lat, fill = bin), width = 0.5, height = 0.5) +
  geom_sf(data = regions, fill = NA, colour = REGION_LINE, linewidth = 0.25) +
  fill_bins + coord_sf(xlim = GLB_X, ylim = GLB_Y, expand = FALSE, datum = NA) + labs(x = NULL, y = NULL) +
  theme_void(base_size = 12) + theme(legend.position = "none", plot.margin = margin(0, 0, 0, 0))
pD <- ggplot() +
  geom_sf(data = world, fill = LAND, colour = BORDER, linewidth = 0.08) +
  geom_tile(data = act_inc, aes(lon, lat, fill = bin), width = 0.5, height = 0.5) +
  geom_sf(data = regions, fill = NA, colour = REGION_LINE, linewidth = 0.25) +
  fill_bins + coord_sf(xlim = AFR_X, ylim = AFR_Y, expand = FALSE, datum = NA) + labs(x = NULL, y = NULL) +
  theme_void(base_size = 12) + theme(legend.position = "none", plot.margin = margin(0, 0, 0, 0))

# legend: one row, labels below keys, title on top (taken from panel A; all panels share the scale)
g_leg <- ggplotGrob(pA + guides(fill = guide_legend(title.position = "top", title.hjust = 0.5, nrow = 1, label.position = "bottom")) +
                      theme(legend.position = "bottom", legend.title = element_text(size = 15, face = "bold"),
                            legend.text = element_text(size = 14), legend.key.width = grid::unit(2.05, "cm"),
                            legend.key.height = grid::unit(0.45, "cm"), legend.spacing.x = grid::unit(0.15, "cm")))
leg <- g_leg$grobs[[which(sapply(g_leg$grobs, function(x) x$name) == "guide-box")]]

# ---- layout (inches): two rows of equal height, global map left, Africa right; titles above, legend at bottom ------
W <- 13; GAP <- 0.2
RA <- diff(GLB_X) / diff(GLB_Y); RB <- diff(AFR_X) / diff(AFR_Y)
HT <- (W - GAP) / (RA + RB); WA <- HT * RA; WB <- HT * RB
TITLE <- 0.45; ROWGAP <- 0.05; LEG <- 1.15
yBot <- LEG                                  # bottom of C and D
yTop <- yBot + HT + TITLE + ROWGAP           # bottom of A and B
H <- yTop + HT + TITLE
stopifnot(WA + GAP + WB <= W + 1e-9)
cat(sprintf("[fig] page %.2f x %.2f in, map height %.2f in\n", W, H, HT))
inch <- function(v) grid::unit(v, "in")
vp <- function(x, y, w, h) grid::viewport(x = inch(x), y = inch(y), width = inch(w), height = inch(h), just = c("left", "bottom"))
tgp <- grid::gpar(fontsize = 19, fontface = "bold")

for (dev in c("png", "pdf")) {
  if (dev == "png") png(f_png, width = W, height = H, units = "in", res = 300, bg = "white")
  if (dev == "pdf") cairo_pdf(f_pdf, width = W, height = H, bg = "white")
  grid::grid.newpage()
  print(pA, vp = vp(0, yTop, WA, HT)); print(pB, vp = vp(WA + GAP, yTop, WB, HT))
  print(pC, vp = vp(0, yBot, WA, HT)); print(pD, vp = vp(WA + GAP, yBot, WB, HT))
  grid::grid.text("A. Base + embedding vs. persistence, global", x = inch(0.03), y = inch(H - 0.05), just = c("left", "top"), gp = tgp)
  grid::grid.text("B. Africa", x = inch(WA + GAP), y = inch(H - 0.05), just = c("left", "top"), gp = tgp)
  grid::grid.text("C. Base + embedding vs. base, global", x = inch(0.03), y = inch(yTop - ROWGAP - 0.05), just = c("left", "top"), gp = tgp)
  grid::grid.text("D. Africa", x = inch(WA + GAP), y = inch(yTop - ROWGAP - 0.05), just = c("left", "top"), gp = tgp)
  grid::pushViewport(vp(0, 0, W, LEG)); grid::grid.draw(leg); grid::popViewport()
  dev.off()
}
stopifnot(file.exists(f_png), file.exists(f_pdf), file.size(f_png) > 0, file.size(f_pdf) > 0)
cat(sprintf("[fig] wrote %s\n[fig] wrote %s\n", f_png, f_pdf))
