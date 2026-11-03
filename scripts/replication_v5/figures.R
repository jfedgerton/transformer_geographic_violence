# figures.R -- every generated figure of the paper in one script (Phase 2 consolidation, 2026-11-03).
# Replaces nine scripts (tag v5-as-run): scripts/phase4_scaling/fig_loro_gains_v2.R, fig_finetune_gains.R,
# fig_placebo_gradient.R, fig_placebo_gradient_a_only_nolabel.R and scripts/replication_v5/fig_cell_maps.R,
# fig_random_encoder.R, fig_share_curve.R, fig_share_regions.R, fig_persistence_gap.R.
# Each block below is the original script's body, unchanged, inside local({ ... }) so blocks cannot see each other's
# variables; only the commandArgs() line is replaced by the exact arguments pipeline/run.sh passed to that script.
# Output folders are created here (the old run.sh did not create fig3_loro_gains).
# Usage (from the project root): Rscript scripts/replication_v5/figures.R <figures_out_dir> <project_root> [fig3,fig4,...]
# Parity: pipeline/tests/parity_phase2.py figures (old scripts vs this file on the same inputs; PNG byte-identical,
# PDF identical apart from creation/modification dates).
args0 <- commandArgs(trailingOnly = TRUE)
if (length(args0) < 2) stop("usage: Rscript figures.R <figures_out_dir> <project_root> [comma-separated figure ids]")
FO <- args0[1]; ROOT_ARG <- args0[2]
stopifnot(dir.exists(ROOT_ARG))
ONLY <- if (length(args0) >= 3) strsplit(args0[3], ",")[[1]] else NULL
NE_ZIP <- "data_raw/shapefiles/ne_50m_admin_0/ne_50m_admin_0_countries.zip"
ALL_IDS <- c("fig3", "fig4", "fig5", "fig6", "fig9", "fig10", "fig16", "fig17", "fig18")
if (!is.null(ONLY)) stopifnot(all(ONLY %in% ALL_IDS))
want <- function(id) is.null(ONLY) || id %in% ONLY

# ===================== fig3: phase4_scaling/fig_loro_gains_v2.R (original body) =====================
if (want("fig3")) {
  dir.create(file.path(FO, "fig3_loro_gains"), recursive = TRUE, showWarnings = FALSE)
  cat("[figures.R] fig3 <- phase4_scaling/fig_loro_gains_v2.R\n")
  local({
# Two-panel LORO figure for Section 4.1. Panel A gain over persistence (base+embedding minus persistence), panel B gain from the embedding (base+embedding minus base).
# Usage Rscript fig_loro_gains.R <bootstrap_ci_csv> <output_dir>
suppressPackageStartupMessages(library(tidyverse))
set.seed(123)
args <- c("outputs/bootstrap_ci_M_fullsrc_block5.csv", file.path(FO, "fig3_loro_gains"))   # was: args <- commandArgs(trailingOnly = TRUE)
infile <- args[1]
outdir <- args[2]
stopifnot(file.exists(infile))
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
d <- read_csv(infile, show_col_types = FALSE) %>% filter(comparison %in% c("persistence", "nested"))
stopifnot(all(c("comparison", "region", "estimate", "lo95", "hi95") %in% names(d)))
stopifnot(nrow(d) == 14)
stopifnot(all(d$lo95 <= d$estimate), all(d$estimate <= d$hi95))
lab <- c(africa = "Africa", americas = "Americas", asia_pacific = "Asia-\nPacific", europe = "Europe", mena = "MENA", south_asia = "South\nAsia", ALL = "All regions")
cols <- c("All regions" = "#000000", "Africa" = "#E69F00", "Americas" = "#56B4E9", "Asia-\nPacific" = "#009E73", "Europe" = "#CC79A7", "MENA" = "#D55E00", "South\nAsia" = "#0072B2")
stopifnot(all(d$region %in% names(lab)))
order_regions <- d %>% filter(comparison == "nested", region %in% setdiff(names(lab), "ALL")) %>% arrange(region) %>% pull(region)
stopifnot(length(order_regions) == 6)
ypos <- c(setNames(6:1, order_regions), ALL = 7.5)
pa <- "A. Gain over persistence (PR-AUC difference)"
pb <- "B. Gain from the embedding (PR-AUC difference)"
d <- d %>% mutate(y = unname(ypos[region]), name = factor(unname(lab[region]), levels = names(cols)), panel = factor(ifelse(comparison == "persistence", pa, pb), levels = c(pa, pb)))
blank <- tibble(panel = factor(pa, levels = c(pa, pb)), x = c(0, 0.40), y = 1)
p <- ggplot(d, aes(y = y, colour = name)) +
  geom_vline(xintercept = 0, linewidth = 1.2, colour = "grey30", linetype = "dashed") +
  geom_hline(yintercept = 6.9, colour = "grey80", linewidth = 0.6) +
  geom_blank(data = blank, aes(x = x, y = y), inherit.aes = FALSE) +
  geom_linerange(aes(xmin = lo95, xmax = hi95), linewidth = 1.9) +
  geom_point(aes(x = estimate), size = 6.5) +
  scale_colour_manual(values = cols, guide = "none") +
  scale_y_continuous(breaks = unname(ypos), labels = unname(lab[names(ypos)]), expand = expansion(add = c(0.6, 0.6))) +
  facet_wrap(~panel, scales = "free_x") +
  labs(x = NULL, y = NULL) +
  theme_minimal(base_size = 16) +
  theme(strip.text = element_text(size = 22, hjust = 0, face = "bold"), axis.text = element_text(size = 19, colour = "black"), panel.grid.major.y = element_blank(), panel.grid.minor = element_blank(), panel.spacing.x = unit(1.4, "cm"), axis.line.x = element_line(colour = "black"))
ggsave(file.path(outdir, "fig_loro_gains.pdf"), p, width = 18, height = 6.2)
ggsave(file.path(outdir, "fig_loro_gains.png"), p, width = 18, height = 6.2, dpi = 300)
f3 <- function(x) sprintf("%.3f", x)
w <- d %>% mutate(txt = paste0(f3(estimate), " [", f3(lo95), ", ", f3(hi95), "]")) %>% select(comparison, region, txt) %>% pivot_wider(names_from = comparison, values_from = txt)
w <- w[match(names(ypos)[order(-ypos)], w$region), ]
stopifnot(nrow(w) == 7, w$region[1] == "ALL")
tex <- c("\\begin{tabular}{lcc}", "\\hline", "Region & Gain over persistence & Gain from embedding \\\\", "\\hline", paste0(gsub("\n", " ", lab[w$region]), " & ", w$persistence, " & ", w$nested, " \\\\"), "\\hline", "\\end{tabular}")
writeLines(tex, file.path(outdir, "tab_loro_gains_ci.tex"))
cat("[done]", file.path(outdir, "fig_loro_gains.pdf"), "\n")
  })
}

# ===================== fig4: replication_v5/fig_cell_maps.R (original body) =====================
if (want("fig4")) {
  dir.create(file.path(FO, "fig4_cell_maps"), recursive = TRUE, showWarnings = FALSE)
  cat("[figures.R] fig4 <- replication_v5/fig_cell_maps.R\n")
  local({
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
args <- c("outputs/cell_alerts_persistence_M_fullsrc.csv.gz", "outputs/cell_alerts_incidence_M_fullsrc.csv.gz", NE_ZIP, file.path(FO, "fig4_cell_maps"))   # was: args <- commandArgs(trailingOnly = TRUE)
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
  })
}

# ===================== fig5: phase4_scaling/fig_finetune_gains.R (original body) =====================
if (want("fig5")) {
  dir.create(file.path(FO, "fig5_finetune_gains"), recursive = TRUE, showWarnings = FALSE)
  cat("[figures.R] fig5 <- phase4_scaling/fig_finetune_gains.R\n")
  local({
# fig_finetune_gains.R -- Results figure for the fine-tuning exercise (step 7b), label fig:finetune_gains.
# Gain in PR-AUC from fine-tuning the pretrained encoder over the same network trained from scratch, by training share.
# Panel A = equal-weight mean of the six held-out regions ("ALL"); Panel B = each held-out region.
# Squares = point estimates (means over 3 encoder seeds x 5 draws); bars = 95% 5x5-degree block bootstrap intervals.
# Filled square = interval excludes zero; hollow = includes zero.
#
# Input : outputs/bootstrap_ft_incidence_block5.csv  (written by scripts/phase4_scaling/bootstrap_ft.py, BOOT_UNIT=block5)
# Output: figures_tables/fig_finetune_gains.pdf and .png (never overwritten; delete or rename the old files to rebuild)
# Run   : Rscript scripts/phase4_scaling/fig_finetune_gains.R [input_csv] [output_dir]
# Added 2026-10-20.
suppressPackageStartupMessages({library(ggplot2); library(dplyr); library(readr)})

ROOT <- "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
args <- c("outputs/bootstrap_ft_fullsrc_M_full_block5.csv", file.path(FO, "fig5_finetune_gains"))   # was: args <- commandArgs(trailingOnly = TRUE)
IN  <- if (length(args) >= 1) args[1] else file.path(ROOT, "outputs/bootstrap_ft_incidence_block5.csv")
OUT <- if (length(args) >= 2) args[2] else file.path(ROOT, "figures_tables")
out_pdf <- file.path(OUT, "fig_finetune_gains.pdf"); out_png <- file.path(OUT, "fig_finetune_gains.png")
stopifnot(file.exists(IN), dir.exists(OUT), !file.exists(out_pdf), !file.exists(out_png))
cat("[in]", IN, "\n")

# ---- 1. read and check ----
d <- read_csv(IN, show_col_types = FALSE)
stopifnot(all(c("target", "comparison", "region", "estimate", "lo95", "hi95", "status") %in% names(d)))
d <- d %>% filter(target == "incidence", grepl("^ft_", comparison))
shares  <- c(0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.1)
regions <- c("ALL", "africa", "americas", "asia_pacific", "europe", "mena", "south_asia")
d <- d %>% mutate(share = as.numeric(sub("^ft_", "", comparison)))
stopifnot(nrow(d) == length(shares) * length(regions),                 # 7 shares x 7 units, no duplicates
          setequal(unique(d$share), shares), setequal(unique(d$region), regions),
          !any(duplicated(d[, c("share", "region")])),
          all(d$status == "ok"),                                       # no 'not estimable' cells
          all(is.finite(d$estimate)), all(d$lo95 <= d$estimate), all(d$hi95 >= d$estimate))

# ---- 2. labels, colors, panels ----
lab <- c(ALL = "All regions", africa = "Africa", americas = "Americas", asia_pacific = "Asia-Pacific",
         europe = "Europe", mena = "MENA", south_asia = "South Asia")
cols <- c("All regions" = "#000000", "Africa" = "#E69F00", "Americas" = "#56B4E9", "Asia-Pacific" = "#009E73",
          "Europe" = "#CC79A7", "MENA" = "#D55E00", "South Asia" = "#0072B2")   # Okabe-Ito, colorblind-safe
d <- d %>% mutate(region = factor(lab[region], levels = lab),
                  share_lab = factor(paste0(share * 100, "%"), levels = paste0(shares * 100, "%")),
                  excl = lo95 > 0 | hi95 < 0,
                  fillv = ifelse(excl, cols[as.character(region)], "white"),
                  panel = factor(ifelse(region == "All regions", "A. All regions", "B. By held-out region"),
                                 levels = c("A. All regions", "B. By held-out region")))
stopifnot(sum(d$panel == "A. All regions") == 7, sum(d$panel == "B. By held-out region") == 42)
print(d %>% select(share, region, estimate, lo95, hi95, excl) %>% arrange(share, region), n = 49)

# ---- 3. plot ----
pd <- position_dodge(width = 0.8)
p <- ggplot(d, aes(share_lab, estimate, colour = region, group = region)) +
  geom_hline(yintercept = 0, colour = "grey55", linewidth = 0.5) +
  geom_linerange(aes(ymin = lo95, ymax = hi95), linewidth = 0.8, position = pd) +
  geom_point(aes(fill = I(fillv)), shape = 22, size = 2.8, stroke = 0.9, position = pd) +
  facet_wrap(~ panel, nrow = 1, scales = "free_x") +
  scale_colour_manual(values = cols, breaks = setdiff(names(cols), "All regions"), name = NULL) +
  guides(colour = guide_legend(nrow = 1, override.aes = list(shape = 22, fill = cols[-1], size = 4, linewidth = 0.8))) +
  labs(x = "Share of the held-out region's 1990 to 2020 rows used for training",
       y = "PR-AUC gain, fine-tuned minus scratch") +
  theme_bw(base_size = 15) +
  theme(panel.grid.minor = element_blank(), panel.grid.major.x = element_blank(),
        axis.text = element_text(size = 13), axis.title = element_text(size = 15),
        strip.background = element_rect(fill = "grey92", colour = NA),
        strip.text = element_text(size = 14, face = "bold", hjust = 0),
        legend.position = "bottom", legend.text = element_text(size = 14),
        legend.key.width = grid::unit(1.2, "lines"))

# Panel B wider than A (six regions per share)
g <- ggplot_gtable(ggplot_build(p))
pcols <- g$layout$l[grepl("^panel", g$layout$name)]
stopifnot(length(pcols) == 2)
g$widths[pcols[1]] <- grid::unit(1.55, "null"); g$widths[pcols[2]] <- grid::unit(2.6, "null")

# ---- 4. save ----
ggsave(out_pdf, g, width = 14, height = 6)
ggsave(out_png, g, width = 14, height = 6, dpi = 300)
stopifnot(file.exists(out_pdf), file.exists(out_png))
cat("[done]", out_pdf, "\n[done]", out_png, "\n")
  })
}

# ===================== fig6: phase4_scaling/fig_placebo_gradient_a_only_nolabel.R (original body) =====================
if (want("fig6")) {
  dir.create(file.path(FO, "fig6_brazil"), recursive = TRUE, showWarnings = FALSE)
  cat("[figures.R] fig6 <- phase4_scaling/fig_placebo_gradient_a_only_nolabel.R\n")
  local({
# fig_placebo_gradient_ab.R -- Brazil cross-domain gradient, BOTH baselines, faceted.
# Panel A = versus the base forecaster (no embedding). Panel B = versus a randomly
# initialized encoder of the same architecture. Pretrained minus baseline, relative
# reduction in Poisson deviance (%), by outcome and training share.
# Filled square = 95% interval excludes zero; hollow = includes zero.
# Input : outputs/crossdomain/xd_placebo_boot_v3M.csv
# Output: outputs/placebo_gradient_ab_M/fig_placebo_gradient_ab_M.{pdf,png} (never overwrites)
# Run   : Rscript scripts/phase4_scaling/fig_placebo_gradient_ab.R [input_csv] [output_dir]
# Added 2026-10-29 by Claude at Jared's request. Sibling of fig_placebo_gradient.R,
# which plots the rand_emb_base baseline only.
suppressPackageStartupMessages({library(ggplot2); library(dplyr); library(readr)})

ROOT <- "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
args <- c("outputs/crossdomain/xd_placebo_boot_v3M.csv", file.path(FO, "fig6_brazil"))   # was: args <- commandArgs(trailingOnly = TRUE)
IN  <- if (length(args) >= 1) args[1] else file.path(ROOT, "outputs/crossdomain/xd_placebo_boot_v3M.csv")
OUT <- if (length(args) >= 2) args[2] else file.path(ROOT, "outputs/placebo_gradient_ab_M")
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
out_pdf <- file.path(OUT, "fig_placebo_gradient_ab_M.pdf")
out_png <- file.path(OUT, "fig_placebo_gradient_ab_M.png")
stopifnot(file.exists(IN), dir.exists(OUT), !file.exists(out_pdf), !file.exists(out_png))
cat("[in]", IN, "\n")

# ---- 1. read and check ----
d <- read_csv(IN, show_col_types = FALSE)
keep  <- c("homicide_self", "suicide", "traffic", "drowning")
bases <- c("base")
d <- d %>% filter(config == "nested_americas_rich_bpaper_capM",
                  baseline %in% bases, model == "emb_base",
                  effect_kind == "relative", placebo %in% keep,
                  status == "complete")
shares <- c(0.0025, 0.01, 0.05, 0.1, 1)
stopifnot(nrow(d) == length(bases) * length(keep) * length(shares),
          setequal(unique(d$share), shares), setequal(unique(d$baseline), bases),
          !any(duplicated(d[, c("placebo", "share", "baseline")])))
d <- d %>% mutate(est = 100 * ifelse(placebo == "homicide_self", effect_homicide, effect_placebo),
                  lo  = 100 * ifelse(placebo == "homicide_self", ci_hom_lo, ci_plac_lo),
                  hi  = 100 * ifelse(placebo == "homicide_self", ci_hom_hi, ci_plac_hi))
stopifnot(all(is.finite(d$est)), all(d$lo <= d$est), all(d$hi >= d$est))

# ---- 2. labels, colors ----
lab  <- c(homicide_self = "Homicide", suicide = "Suicide", traffic = "Traffic deaths", drowning = "Drowning")
cols <- c("Homicide" = "#D55E00", "Suicide" = "#0072B2", "Traffic deaths" = "#009E73", "Drowning" = "#E69F00")
panlab <- c(base = "Versus the base forecaster",
            rand_emb_base = "B. Versus a randomly initialized encoder")
d <- d %>% mutate(outcome = factor(lab[placebo], levels = lab),
                  share_lab = factor(paste0(share * 100, "%"), levels = paste0(shares * 100, "%")),
                  excl = lo > 0 | hi < 0,
                  fillv = ifelse(excl, cols[as.character(outcome)], "white"),
                  panel = factor(panlab[baseline], levels = panlab))
print(as.data.frame(d %>% select(panel, outcome, share, est, lo, hi, excl) %>% arrange(panel, share, outcome)))

# ---- 3. plot ----
pd <- position_dodge(width = 0.6)
p <- ggplot(d, aes(share_lab, est, colour = outcome, group = outcome)) +
  geom_hline(yintercept = 0, colour = "grey55", linewidth = 0.5) +
  geom_linerange(aes(ymin = lo, ymax = hi), linewidth = 0.8, position = pd) +
  geom_point(aes(fill = I(fillv)), shape = 22, size = 2.8, stroke = 0.9, position = pd) +
  scale_colour_manual(values = cols, name = NULL) +
  guides(colour = guide_legend(nrow = 1, override.aes = list(shape = 22, fill = cols, size = 4, linewidth = 0.8))) +
  facet_wrap(~ panel, nrow = 1) +
  labs(x = "Share of Brazil's training data used for the forecaster",
       y = "Deviance reduction (%)") +
  theme_bw(base_size = 15) +
  theme(panel.grid.minor = element_blank(), panel.grid.major.x = element_blank(),
        axis.text = element_text(size = 13), axis.title = element_text(size = 15),
        strip.text = element_text(size = 14, face = "bold", hjust = 0),
        strip.background = element_rect(fill = "grey92"),
        panel.spacing = grid::unit(1.2, "lines"),
        legend.position = "bottom", legend.text = element_text(size = 14),
        legend.key.width = grid::unit(1.2, "lines"))

# ---- 4. save ----
ggsave(out_pdf, p, width = 7, height = 5.5)
ggsave(out_png, p, width = 7, height = 5.5, dpi = 300)
stopifnot(file.exists(out_pdf), file.exists(out_png))
cat("[done]", out_pdf, "\n[done]", out_png, "\n")
  })
}

# ===================== fig9: replication_v5/fig_random_encoder.R (original body) =====================
if (want("fig9")) {
  dir.create(file.path(FO, "fig9_random_encoder"), recursive = TRUE, showWarnings = FALSE)
  cat("[figures.R] fig9 <- replication_v5/fig_random_encoder.R\n")
  local({
# fig_random_encoder.R -- Appendix B figure, label fig:random (compiled Figure 9).
# Gain over the base model (base + embedding minus base, PR-AUC, incidence) from the pretrained and the
# random-weight M full-source encoder, by held-out region; temporal test (B_temporal) and pooled test (mean of A_pooled_f0-f4).
# Means over three encoder seeds. Same inputs as tables tab:random_temporal and tab:random_pooled.
#
# Input : outputs/eval_region_M_contrastive_e_fullsrc_loro_{region}_s{seed}.csv       (pretrained)
#         outputs/eval_region_M_contrastive_e_fullsrc_loro_{region}_s{seed}_rand.csv  (random weights)
#         written by scripts/phase4_scaling/transfer_eval_region_fullpreds.py (launch_fullsrc_eval{,_rand}.sbatch)
# Output: <out_dir>/fig_random_encoder.pdf and .png (never overwritten)
# Run   : FIG_ROOT=<conflict_forecast dir> Rscript fig_random_encoder.R <out_dir> [root]
suppressPackageStartupMessages({library(ggplot2); library(dplyr); library(readr); library(tidyr)})

args <- c(file.path(FO, "fig9_random_encoder"), ROOT_ARG)   # was: args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) >= 1)
OUT  <- args[1]
ROOT <- if (length(args) >= 2) args[2] else Sys.getenv("FIG_ROOT", "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast")
out_pdf <- file.path(OUT, "fig_random_encoder.pdf"); out_png <- file.path(OUT, "fig_random_encoder.png")
stopifnot(dir.exists(ROOT), dir.exists(OUT), !file.exists(out_pdf), !file.exists(out_png))

# ---- 1. read the 36 eval files (6 regions x 3 seeds x 2 encoders) ----
regions <- c("africa", "americas", "asia_pacific", "europe", "mena", "south_asia")
seeds   <- c("s123", "s456", "s789")
modes   <- c("B_temporal", paste0("A_pooled_f", 0:4))
rows <- list()
for (r in regions) for (s in seeds) for (enc in c("pretrained", "random")) {
  f <- file.path(ROOT, "outputs", paste0("eval_region_M_contrastive_e_fullsrc_loro_", r, "_", s,
                                         ifelse(enc == "random", "_rand", ""), ".csv"))
  stopifnot(file.exists(f))
  x <- read_csv(f, show_col_types = FALSE)
  stopifnot(all(c("unit", "mode", "target", "model", "score") %in% names(x)), all(x$unit == r))
  x <- x %>% filter(target == "incidence", mode %in% modes, model %in% c("base", "emb_base"))
  stopifnot(nrow(x) == length(modes) * 2, !any(duplicated(x[, c("mode", "model")])), all(is.finite(x$score)))
  rows[[length(rows) + 1]] <- x %>% mutate(region = r, seed = s, encoder = enc)
}
d <- bind_rows(rows)
stopifnot(nrow(d) == 6 * 3 * 2 * 6 * 2)

# base model is identical across the two arms (same base fit, same test rows)
chk <- d %>% filter(model == "base") %>% select(region, seed, mode, encoder, score) %>%
  pivot_wider(names_from = encoder, values_from = score)
stopifnot(nrow(chk) == 6 * 3 * 6, max(abs(chk$pretrained - chk$random)) < 1e-9)

# ---- 2. gains: per region-seed-mode, then pooled = mean over 5 folds, then mean over seeds ----
g <- d %>% select(region, seed, encoder, mode, model, score) %>%
  pivot_wider(names_from = model, values_from = score) %>%
  mutate(gain = emb_base - base, test = ifelse(mode == "B_temporal", "Temporal", "Pooled")) %>%
  group_by(region, seed, encoder, test) %>%
  summarise(n_modes = n(), gain = mean(gain), .groups = "drop")
stopifnot(all(g$n_modes[g$test == "Temporal"] == 1), all(g$n_modes[g$test == "Pooled"] == 5))

runs <- g %>% select(region, seed, encoder, test, gain) %>% pivot_wider(names_from = encoder, values_from = gain) %>%
  group_by(region, test) %>% summarise(runs_pos = sum(pretrained - random > 0), n_runs = n(), .groups = "drop")
stopifnot(all(runs$n_runs == 3))
m <- g %>% group_by(region, encoder, test) %>% summarise(n_seeds = n(), gain = mean(gain), .groups = "drop")
stopifnot(nrow(m) == 6 * 2 * 2, all(m$n_seeds == 3))

# printout for cross-checking against tab:random_temporal / tab:random_pooled
w <- m %>% select(-n_seeds) %>% pivot_wider(names_from = encoder, values_from = gain) %>%
  mutate(diff = pretrained - random) %>% left_join(runs, by = c("region", "test")) %>% arrange(test, region)
print(w %>% mutate(across(c(pretrained, random, diff), ~ sprintf("%+.3f", .x))), n = 12)
print(w %>% group_by(test) %>% summarise(pretrained = sprintf("%+.3f", mean(pretrained)), random = sprintf("%+.3f", mean(random)),
                                         diff = sprintf("%+.3f", mean(diff)), runs = paste0(sum(runs_pos), "/", sum(n_runs))))

# ---- 3. plot ----
lab <- c(africa = "Africa", americas = "Americas", asia_pacific = "Asia-Pacific", europe = "Europe",
         mena = "MENA", south_asia = "South Asia")
m <- m %>% mutate(region = factor(lab[region], levels = rev(lab)),
                  test = factor(test, levels = c("Temporal", "Pooled")),
                  encoder = factor(ifelse(encoder == "pretrained", "Pretrained encoder", "Random encoder"),
                                   levels = c("Pretrained encoder", "Random encoder")))
seg <- m %>% select(region, test, encoder, gain) %>% pivot_wider(names_from = encoder, values_from = gain)
stopifnot(nrow(seg) == 12)

p <- ggplot(m, aes(y = region)) +
  geom_vline(xintercept = 0, colour = "grey55", linewidth = 0.5) +
  geom_segment(data = seg, aes(x = `Random encoder`, xend = `Pretrained encoder`, y = region, yend = region),
               colour = "grey80", linewidth = 2.2) +
  geom_point(aes(x = gain, colour = encoder, shape = encoder), size = 4.2) +
  facet_wrap(~ test, nrow = 1) +
  scale_colour_manual(values = c("Pretrained encoder" = "#009E73", "Random encoder" = "#D55E00"), name = NULL) +
  scale_shape_manual(values = c("Pretrained encoder" = 15, "Random encoder" = 18), name = NULL) +
  scale_x_continuous(labels = function(x) sprintf("%+.2f", x)) +
  labs(x = "PR-AUC gain over the base model (base + embedding minus base)", y = NULL) +
  theme_bw(base_size = 15) +
  theme(panel.grid.minor = element_blank(), panel.grid.major.y = element_blank(),
        axis.text = element_text(size = 13, colour = "black"), axis.title = element_text(size = 15),
        strip.background = element_rect(fill = "grey92", colour = NA),
        strip.text = element_text(size = 14, face = "bold", hjust = 0),
        legend.position = "bottom", legend.text = element_text(size = 14, margin = margin(r = 18)),
        panel.spacing.x = grid::unit(1.2, "lines"))

# ---- 4. save ----
ggsave(out_pdf, p, width = 12, height = 5.5)
ggsave(out_png, p, width = 12, height = 5.5, dpi = 300)
stopifnot(file.exists(out_pdf), file.exists(out_png))
cat("[done]", out_pdf, "\n[done]", out_png, "\n")
  })
}

# ===================== fig10: replication_v5/fig_share_curve.R (original body) =====================
if (want("fig10")) {
  dir.create(file.path(FO, "fig10_share_curve"), recursive = TRUE, showWarnings = FALSE)
  cat("[figures.R] fig10 <- replication_v5/fig_share_curve.R\n")
  local({
# fig_share_curve.R -- Appendix C figure, label fig:boot_shares (compiled Figure 10).
# Gain from adding the frozen M full-source embedding to the base model (base + embedding minus base, PR-AUC,
# temporal test) when the forecaster sees 0.25-10% of the held-out region's training data.
# Teal line + band = equal-weight mean of the six regions ("ALL") with its 95% 5x5-degree block bootstrap interval;
# gray lines = individual held-out regions (point estimates). Estimates are means over 3 seeds x 5 draws.
# Same input as tables tab:pretrained_1pct and tab:boot_shares.
#
# Input : outputs/bootstrap_ci_M_fullsrc_block5_shares.csv (scripts/phase4_scaling/bootstrap_ci_fullsrc.py,
#         launch_bootstrap_shares_fullsrc.sbatch, BOOT_SIZE=M)
# Output: <out_dir>/fig_boot_shares.pdf and .png (never overwritten)
# Run   : FIG_ROOT=<conflict_forecast dir> Rscript fig_share_curve.R <out_dir> [root]
suppressPackageStartupMessages({library(ggplot2); library(dplyr); library(readr)})

args <- c(file.path(FO, "fig10_share_curve"), ROOT_ARG)   # was: args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) >= 1)
OUT  <- args[1]
ROOT <- if (length(args) >= 2) args[2] else Sys.getenv("FIG_ROOT", "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast")
IN <- file.path(ROOT, "outputs/bootstrap_ci_M_fullsrc_block5_shares.csv")
out_pdf <- file.path(OUT, "fig_boot_shares.pdf"); out_png <- file.path(OUT, "fig_boot_shares.png")
stopifnot(file.exists(IN), dir.exists(OUT), !file.exists(out_pdf), !file.exists(out_png))
cat("[in]", IN, "\n")

# ---- 1. read and check ----
d <- read_csv(IN, show_col_types = FALSE)
stopifnot(all(c("comparison", "region", "estimate", "lo95", "hi95", "B") %in% names(d)))
shares  <- c(0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.1)
regions <- c("ALL", "africa", "americas", "asia_pacific", "europe", "mena", "south_asia")
d <- d %>% mutate(share = as.numeric(sub("^share_", "", comparison)))
stopifnot(all(grepl("^share_", d$comparison)), nrow(d) == 49,
          setequal(unique(d$share), shares), setequal(unique(d$region), regions),
          !any(duplicated(d[, c("share", "region")])), all(d$B == 1000),
          all(is.finite(d$estimate)), all(d$lo95 <= d$estimate), all(d$hi95 >= d$estimate))
# ALL must be the equal-weight mean of the six regions
chk <- d %>% filter(region != "ALL") %>% group_by(share) %>% summarise(m = mean(estimate)) %>%
  inner_join(d %>% filter(region == "ALL") %>% select(share, estimate), by = "share")
stopifnot(nrow(chk) == 7, max(abs(chk$m - chk$estimate)) < 1e-9)

d <- d %>% mutate(x = match(share, shares))     # equal spacing, as in fig_finetune_gains.R
print(d %>% filter(region == "ALL") %>% arrange(share) %>%
        transmute(share, est = sprintf("%+.3f", estimate), ci = sprintf("[%+.3f, %+.3f]", lo95, hi95)))
print(d %>% filter(region != "ALL") %>% group_by(share) %>% summarise(regions_pos = sum(estimate > 0)))

# ---- 2. plot ----
mean_d <- d %>% filter(region == "ALL")
reg_d  <- d %>% filter(region != "ALL")
p <- ggplot() +
  geom_hline(yintercept = 0, colour = "grey55", linewidth = 0.5) +
  geom_line(data = reg_d, aes(x, estimate, group = region), colour = "grey65", linewidth = 0.7) +
  geom_ribbon(data = mean_d, aes(x, ymin = lo95, ymax = hi95), fill = "#009E73", alpha = 0.25) +
  geom_line(data = mean_d, aes(x, estimate), colour = "#009E73", linewidth = 1.6) +
  geom_point(data = mean_d, aes(x, estimate), colour = "#009E73", shape = 15, size = 3.2) +
  scale_x_continuous(breaks = seq_along(shares), labels = paste0(shares * 100, "%"),
                     expand = expansion(add = 0.3)) +
  scale_y_continuous(breaks = function(l) pretty(l, n = 6), labels = function(x) sprintf("%+.3f", x)) +
  labs(x = "Share of the held-out region's training data", y = "PR-AUC gain from the embedding") +
  theme_bw(base_size = 15) +
  theme(panel.grid.minor = element_blank(), panel.grid.major.x = element_blank(),
        axis.text = element_text(size = 13, colour = "black"), axis.title = element_text(size = 15))

# ---- 3. save ----
ggsave(out_pdf, p, width = 10, height = 5.5)
ggsave(out_png, p, width = 10, height = 5.5, dpi = 300)
stopifnot(file.exists(out_pdf), file.exists(out_png))
cat("[done]", out_pdf, "\n[done]", out_png, "\n")
  })
}

# ===================== fig16: phase4_scaling/fig_placebo_gradient.R (original body) =====================
if (want("fig16")) {
  dir.create(file.path(FO, "fig16_brazil_random"), recursive = TRUE, showWarnings = FALSE)
  cat("[figures.R] fig16 <- phase4_scaling/fig_placebo_gradient.R\n")
  local({
# fig_placebo_gradient.R -- Placebo gradient for the Brazil cross-domain test, label fig:placebo_gradient.
# Pretrained minus random encoder, relative reduction in Poisson deviance (%), by outcome and training share.
# Filled square = 95% interval excludes zero; hollow = includes zero.
#
# Input : outputs/crossdomain/xd_placebo_boot_v3M.csv (M Americas-held-out encoder, config nested_americas_rich_bpaper_capM)
# Output: outputs/placebo_gradient_M/fig_placebo_gradient_M.pdf and .png (never overwritten)
# Run   : Rscript scripts/phase4_scaling/fig_placebo_gradient.R [input_csv] [output_dir]
# Added 2026-10-29 by Claude at Jared's request.
suppressPackageStartupMessages({library(ggplot2); library(dplyr); library(readr)})

ROOT <- "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
args <- c("outputs/crossdomain/xd_placebo_boot_v3M.csv", file.path(FO, "fig16_brazil_random"))   # was: args <- commandArgs(trailingOnly = TRUE)
IN  <- if (length(args) >= 1) args[1] else file.path(ROOT, "outputs/crossdomain/xd_placebo_boot_v3M.csv")
OUT <- if (length(args) >= 2) args[2] else file.path(ROOT, "outputs/placebo_gradient_M")
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
out_pdf <- file.path(OUT, "fig_placebo_gradient_M.pdf"); out_png <- file.path(OUT, "fig_placebo_gradient_M.png")
stopifnot(file.exists(IN), dir.exists(OUT), !file.exists(out_pdf), !file.exists(out_png))
cat("[in]", IN, "\n")

# ---- 1. read and check ----
d <- read_csv(IN, show_col_types = FALSE)
keep <- c("homicide_self", "suicide", "traffic", "drowning")
d <- d %>% filter(config == "nested_americas_rich_bpaper_capM",
                  baseline == "rand_emb_base", model == "emb_base",
                  effect_kind == "relative", placebo %in% keep)
shares <- c(0.0025, 0.01, 0.05, 0.1, 1)
stopifnot(nrow(d) == length(keep) * length(shares), all(d$status == "complete"),
          setequal(unique(d$share), shares), !any(duplicated(d[, c("placebo", "share")])))
d <- d %>% mutate(est = 100 * ifelse(placebo == "homicide_self", effect_homicide, effect_placebo),
                  lo  = 100 * ifelse(placebo == "homicide_self", ci_hom_lo, ci_plac_lo),
                  hi  = 100 * ifelse(placebo == "homicide_self", ci_hom_hi, ci_plac_hi))
stopifnot(all(is.finite(d$est)), all(d$lo <= d$est), all(d$hi >= d$est))

# ---- 2. labels, colors ----
lab <- c(homicide_self = "Homicide", suicide = "Suicide", traffic = "Traffic deaths", drowning = "Drowning")
cols <- c("Homicide" = "#D55E00", "Suicide" = "#0072B2", "Traffic deaths" = "#009E73", "Drowning" = "#E69F00")
d <- d %>% mutate(outcome = factor(lab[placebo], levels = lab),
                  share_lab = factor(paste0(share * 100, "%"), levels = paste0(shares * 100, "%")),
                  excl = lo > 0 | hi < 0,
                  fillv = ifelse(excl, cols[as.character(outcome)], "white"))
print(as.data.frame(d %>% select(outcome, share, est, lo, hi, excl) %>% arrange(share, outcome)))

# ---- 3. plot ----
pd <- position_dodge(width = 0.6)
p <- ggplot(d, aes(share_lab, est, colour = outcome, group = outcome)) +
  geom_hline(yintercept = 0, colour = "grey55", linewidth = 0.5) +
  geom_linerange(aes(ymin = lo, ymax = hi), linewidth = 0.8, position = pd) +
  geom_point(aes(fill = I(fillv)), shape = 22, size = 2.8, stroke = 0.9, position = pd) +
  scale_colour_manual(values = cols, name = NULL) +
  guides(colour = guide_legend(nrow = 1, override.aes = list(shape = 22, fill = cols, size = 4, linewidth = 0.8))) +
  labs(x = "Share of Brazil's training data used for the forecaster",
       y = "Deviance reduction, pretrained minus random encoder (%)") +
  theme_bw(base_size = 15) +
  theme(panel.grid.minor = element_blank(), panel.grid.major.x = element_blank(),
        axis.text = element_text(size = 13), axis.title = element_text(size = 15),
        legend.position = "bottom", legend.text = element_text(size = 14),
        legend.key.width = grid::unit(1.2, "lines"))

# ---- 4. save ----
ggsave(out_pdf, p, width = 10, height = 5.5)
ggsave(out_png, p, width = 10, height = 5.5, dpi = 300)
stopifnot(file.exists(out_pdf), file.exists(out_png))
cat("[done]", out_pdf, "\n[done]", out_png, "\n")
  })
}

# ===================== fig17: replication_v5/fig_share_regions.R (original body) =====================
if (want("fig17")) {
  dir.create(file.path(FO, "fig17_share_regions"), recursive = TRUE, showWarnings = FALSE)
  cat("[figures.R] fig17 <- replication_v5/fig_share_regions.R\n")
  local({
# fig_share_regions.R -- Appendix bootstrap-section figure, label fig:boot_share_regions (compiled Figure 17).
# Gain from adding the frozen M full-source embedding (base + embedding minus base, PR-AUC, temporal test) by
# training share, one panel per held-out region. Band = 95% 5x5-degree block bootstrap interval;
# vertical lines = 95% single-cell bootstrap interval; squares = point estimates (3 seeds x 5 draws). Free y-axes.
# Same input as table tab:boot_shares.
#
# Input : outputs/bootstrap_ci_M_fullsrc_block5_shares.csv and outputs/bootstrap_ci_M_fullsrc_cell_shares.csv
#         (scripts/phase4_scaling/bootstrap_ci_fullsrc.py, launch_bootstrap_shares_fullsrc.sbatch, BOOT_SIZE=M)
# Output: <out_dir>/fig_boot_share_regions.pdf and .png (never overwritten)
# Run   : FIG_ROOT=<conflict_forecast dir> Rscript fig_share_regions.R <out_dir> [root]
suppressPackageStartupMessages({library(ggplot2); library(dplyr); library(readr)})

args <- c(file.path(FO, "fig17_share_regions"), ROOT_ARG)   # was: args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) >= 1)
OUT  <- args[1]
ROOT <- if (length(args) >= 2) args[2] else Sys.getenv("FIG_ROOT", "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast")
IN_B <- file.path(ROOT, "outputs/bootstrap_ci_M_fullsrc_block5_shares.csv")
IN_C <- file.path(ROOT, "outputs/bootstrap_ci_M_fullsrc_cell_shares.csv")
out_pdf <- file.path(OUT, "fig_boot_share_regions.pdf"); out_png <- file.path(OUT, "fig_boot_share_regions.png")
stopifnot(file.exists(IN_B), file.exists(IN_C), dir.exists(OUT), !file.exists(out_pdf), !file.exists(out_png))

# ---- 1. read and check ----
shares  <- c(0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.1)
regions <- c("africa", "americas", "asia_pacific", "europe", "mena", "south_asia")
rd <- function(f) {
  x <- read_csv(f, show_col_types = FALSE)
  stopifnot(all(c("comparison", "region", "estimate", "lo95", "hi95", "B") %in% names(x)),
            all(grepl("^share_", x$comparison)), nrow(x) == 49, all(x$B == 1000))
  x <- x %>% mutate(share = as.numeric(sub("^share_", "", comparison))) %>% filter(region %in% regions)
  stopifnot(nrow(x) == 42, setequal(unique(x$share), shares), setequal(unique(x$region), regions),
            !any(duplicated(x[, c("share", "region")])),
            all(is.finite(x$estimate)), all(x$lo95 <= x$estimate), all(x$hi95 >= x$estimate))
  x %>% select(share, region, estimate, lo95, hi95)
}
b <- rd(IN_B); cc <- rd(IN_C)
d <- b %>% inner_join(cc, by = c("share", "region"), suffix = c("", "_cell"))
stopifnot(nrow(d) == 42, max(abs(d$estimate - d$estimate_cell)) < 1e-12)   # same point estimates in both files
d <- d %>% mutate(x = match(share, shares))
print(d %>% arrange(region, share) %>%
        transmute(region, share, est = sprintf("%+.3f", estimate), block = sprintf("[%+.3f, %+.3f]", lo95, hi95),
                  cell = sprintf("[%+.3f, %+.3f]", lo95_cell, hi95_cell)), n = 42)

# ---- 2. labels, colors ----
lab  <- c(africa = "Africa", americas = "Americas", asia_pacific = "Asia-Pacific", europe = "Europe",
          mena = "MENA", south_asia = "South Asia")
cols <- c("Africa" = "#E69F00", "Americas" = "#56B4E9", "Asia-Pacific" = "#009E73",
          "Europe" = "#CC79A7", "MENA" = "#D55E00", "South Asia" = "#0072B2")   # Okabe-Ito, as fig_finetune_gains.R
d <- d %>% mutate(region = factor(lab[region], levels = lab))

# ---- 3. plot ----
p <- ggplot(d, aes(x, estimate, colour = region, fill = region)) +
  geom_hline(yintercept = 0, colour = "grey55", linewidth = 0.5) +
  geom_ribbon(aes(ymin = lo95, ymax = hi95), alpha = 0.22, colour = NA) +
  geom_linerange(aes(ymin = lo95_cell, ymax = hi95_cell), linewidth = 0.8) +
  geom_line(linewidth = 1.1) +
  geom_point(shape = 15, size = 2.6) +
  facet_wrap(~ region, nrow = 2, scales = "free_y") +
  scale_colour_manual(values = cols, guide = "none") +
  scale_fill_manual(values = cols, guide = "none") +
  scale_x_continuous(breaks = seq_along(shares), labels = paste0(shares * 100), expand = expansion(add = 0.3)) +
  scale_y_continuous(breaks = function(l) pretty(l, n = 5), labels = function(x) sprintf("%+.3f", x)) +
  labs(x = "Share of the held-out region's training data (%)", y = "PR-AUC gain from the embedding") +
  theme_bw(base_size = 15) +
  theme(panel.grid.minor = element_blank(), panel.grid.major.x = element_blank(),
        axis.text = element_text(size = 12, colour = "black"), axis.title = element_text(size = 15),
        strip.background = element_rect(fill = "grey92", colour = NA),
        strip.text = element_text(size = 14, face = "bold", hjust = 0),
        panel.spacing = grid::unit(1, "lines"))

# ---- 4. save ----
ggsave(out_pdf, p, width = 13, height = 8)
ggsave(out_png, p, width = 13, height = 8, dpi = 300)
stopifnot(file.exists(out_pdf), file.exists(out_png))
cat("[done]", out_pdf, "\n[done]", out_png, "\n")
  })
}

# ===================== fig18: replication_v5/fig_persistence_gap.R (original body) =====================
if (want("fig18")) {
  dir.create(file.path(FO, "fig18_persistence_gap"), recursive = TRUE, showWarnings = FALSE)
  cat("[figures.R] fig18 <- replication_v5/fig_persistence_gap.R\n")
  local({
# fig_persistence_gap.R -- Appendix bootstrap-section figure, label fig:boot_persistence (compiled Figure 18).
# Base model with the M full-source embedding minus month-t persistence (PR-AUC, temporal test), by held-out region
# and the equal-weight mean ("All regions"). Wide bars = 95% 5x5-degree block bootstrap interval; thin bars = 95%
# single-cell interval; squares = means over three encoder seeds. Same input as table tab:boot_main panel A.
#
# Input : outputs/bootstrap_ci_M_fullsrc_block5.csv and outputs/bootstrap_ci_M_fullsrc_cell.csv
#         (scripts/phase4_scaling/bootstrap_ci_fullsrc.py, launch_bootstrap_ci_fullsrc.sbatch, BOOT_SIZE=M)
# Output: <out_dir>/fig_boot_persistence.pdf and .png (never overwritten)
# Run   : FIG_ROOT=<conflict_forecast dir> Rscript fig_persistence_gap.R <out_dir> [root]
suppressPackageStartupMessages({library(ggplot2); library(dplyr); library(readr)})

args <- c(file.path(FO, "fig18_persistence_gap"), ROOT_ARG)   # was: args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) >= 1)
OUT  <- args[1]
ROOT <- if (length(args) >= 2) args[2] else Sys.getenv("FIG_ROOT", "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast")
IN_B <- file.path(ROOT, "outputs/bootstrap_ci_M_fullsrc_block5.csv")
IN_C <- file.path(ROOT, "outputs/bootstrap_ci_M_fullsrc_cell.csv")
out_pdf <- file.path(OUT, "fig_boot_persistence.pdf"); out_png <- file.path(OUT, "fig_boot_persistence.png")
stopifnot(file.exists(IN_B), file.exists(IN_C), dir.exists(OUT), !file.exists(out_pdf), !file.exists(out_png))

# ---- 1. read and check ----
regions <- c("africa", "americas", "asia_pacific", "europe", "mena", "south_asia", "ALL")
rd <- function(f) {
  x <- read_csv(f, show_col_types = FALSE)
  stopifnot(all(c("comparison", "region", "estimate", "lo95", "hi95", "B") %in% names(x)), all(x$B == 1000))
  x <- x %>% filter(comparison == "persistence")
  stopifnot(nrow(x) == 7, setequal(x$region, regions), !any(duplicated(x$region)),
            all(is.finite(x$estimate)), all(x$lo95 <= x$estimate), all(x$hi95 >= x$estimate))
  x %>% select(region, estimate, lo95, hi95)
}
b <- rd(IN_B); cc <- rd(IN_C)
d <- b %>% inner_join(cc, by = "region", suffix = c("", "_cell"))
stopifnot(nrow(d) == 7, max(abs(d$estimate - d$estimate_cell)) < 1e-12)
stopifnot(abs(mean(d$estimate[d$region != "ALL"]) - d$estimate[d$region == "ALL"]) < 1e-9)   # ALL = equal-weight mean
print(d %>% transmute(region, est = sprintf("%+.3f", estimate), block = sprintf("[%+.3f, %+.3f]", lo95, hi95),
                      cell = sprintf("[%+.3f, %+.3f]", lo95_cell, hi95_cell)))

# ---- 2. labels, colors ----
lab  <- c(africa = "Africa", americas = "Americas", asia_pacific = "Asia-Pacific", europe = "Europe",
          mena = "MENA", south_asia = "South Asia", ALL = "All regions")
cols <- c("All regions" = "#000000", "Africa" = "#E69F00", "Americas" = "#56B4E9", "Asia-Pacific" = "#009E73",
          "Europe" = "#CC79A7", "MENA" = "#D55E00", "South Asia" = "#0072B2")   # Okabe-Ito, as fig_finetune_gains.R
d <- d %>% mutate(name = factor(lab[region], levels = rev(lab)))   # All regions at the bottom

# ---- 3. plot ----
p <- ggplot(d, aes(y = name, colour = name)) +
  geom_linerange(aes(xmin = lo95, xmax = hi95), linewidth = 6, alpha = 0.35) +
  geom_linerange(aes(xmin = lo95_cell, xmax = hi95_cell), linewidth = 1.2) +
  geom_point(aes(x = estimate), shape = 15, size = 4) +
  scale_colour_manual(values = cols, guide = "none") +
  scale_x_continuous(breaks = function(l) pretty(l, n = 6), labels = function(x) sprintf("%+.2f", x),
                     expand = expansion(mult = c(0.04, 0.06))) +
  coord_cartesian(clip = "off") +
  labs(x = "Base model with embedding minus persistence (PR-AUC, temporal test)", y = NULL) +
  theme_bw(base_size = 15) +
  theme(panel.grid.minor = element_blank(), panel.grid.major.y = element_blank(),
        axis.text = element_text(size = 13, colour = "black"), axis.title = element_text(size = 15),
        plot.margin = margin(8, 22, 8, 8))

# ---- 4. save ----
ggsave(out_pdf, p, width = 10, height = 5.5)
ggsave(out_png, p, width = 10, height = 5.5, dpi = 300)
stopifnot(file.exists(out_pdf), file.exists(out_png))
cat("[done]", out_pdf, "\n[done]", out_png, "\n")
  })
}
cat("[figures.R] done\n")
