# figures.R -- every generated figure of the paper in one script (Phase 2 consolidation, 2026-11-03).
# Replaces nine scripts (tag v5-as-run): scripts/phase4_scaling/fig_loro_gains_v2.R, fig_finetune_gains.R,
# fig_placebo_gradient.R, fig_placebo_gradient_a_only_nolabel.R and scripts/replication_v5/fig_cell_maps.R,
# fig_random_encoder.R, fig_share_curve.R, fig_share_regions.R, fig_persistence_gap.R.
# Each block below is the original script's body, unchanged, inside local({ ... }) so blocks cannot see each other's
# variables; only the commandArgs() line is replaced by the exact arguments pipeline/run.sh passed to that script.
# Output folders are created here (the old run.sh did not create fig3_loro_gains).
# Usage (from the project root): Rscript scripts/replication_v5/figures.R <figures_out_dir> <project_root> [fig3,fig4,...]
# Single-figure mode (2026-11-05) for the three figures built in earlier stages, with that script's own arguments:
#   Rscript figures.R --one fig19 <cell_alerts_csv.gz> <ne_50m_zip> <target> <out.png>     (was figures_tables/fig_cell_alerts.R)
#   Rscript figures.R --one ukr_map <cells_csv> [shapefile] [out_dir]                      (was scripts/phase4_scaling/fig_ukraine_map.R)
#   UKR_ARM=top2|full Rscript figures.R --one ukr_gain <csv_0.0025> <csv_0.1> <out_dir> [shp] (was replication_v5/fig_ukraine_alert_gain_arm.R)
# In this mode only that block runs, in its own R session as before; its commandArgs() line is replaced by the passed args.
# Parity: pipeline/tests/parity_phase2.py figures (old scripts vs this file on the same inputs; PNG byte-identical,
# PDF identical apart from creation/modification dates).
args0 <- commandArgs(trailingOnly = TRUE)
ONE_IDS <- c("fig19", "ukr_map", "ukr_gain")
if (length(args0) >= 2 && args0[1] == "--one") {
  ONE <- args0[2]; PASS <- args0[-(1:2)]
  if (!(ONE %in% ONE_IDS)) stop("--one needs one of: ", paste(ONE_IDS, collapse = ", "))
  ONLY <- character(0)   # none of the nine stage-09 figures
} else {
ONE <- ""
if (length(args0) < 2) stop("usage: Rscript figures.R <figures_out_dir> <project_root> [comma-separated figure ids]")
FO <- args0[1]; ROOT_ARG <- args0[2]
stopifnot(dir.exists(ROOT_ARG))
ONLY <- if (length(args0) >= 3) strsplit(args0[3], ",")[[1]] else NULL
}
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
# fig_cell_maps.R -- main-text Figure 4 (fig:cell_maps): two 0.5-degree cell maps of alerts changed by the embedding.
# A = base + embedding vs. base (nested), global    B = same, Africa
# 2026-11-05 (Jared): the persistence row (old panels A-B, base + embedding vs. persistence) is dropped from the paper.
# Each cell: correct alerts gained minus false alerts added (net = d_tp - d_fp), already averaged over the three encoder
# seeds in the input files (cell_alerts_agg.py output; one row per PRIO-GRID cell). Conventions (palette, bins, cell
# geometry, Natural Earth borders, held-out region outlines, map extents) copied from figures_tables/fig_cell_alerts.R.
# Old four-panel version: figures/fig_cell_alerts_maps_pers_nested_v2.png (original script fig_maps_2row.R is lost).
#
# Note on totals: at equal alert budgets every extra correct alert displaces one false alert, so sum(d_fp) = -sum(d_tp)
# and sum(net) = 2 * sum(d_tp). The numbers in Sec. 4.1 (710 / 197) are the extra correct alerts,
# sum(d_tp) = sum(net) / 2; the script checks both.
#
# Usage: Rscript fig_cell_maps.R <incidence_csv_gz> <ne_zip> <out_dir>
# Tested with R 4.3.3, ggplot2 3.4.4, sf 1.0.15.
set.seed(123)
for (p in c("tidyverse", "sf")) if (!requireNamespace(p, quietly = TRUE)) stop("R package ", p, " is not installed")
suppressPackageStartupMessages({ library(tidyverse); library(sf) })
sf_use_s2(FALSE)
# UTF-8 locale so the degree sign renders in the legend title (a C locale prints it as "..")
if (!l10n_info()$`UTF-8`) invisible(suppressWarnings(Sys.setlocale("LC_CTYPE", "C.UTF-8")))
stopifnot(l10n_info()$`UTF-8`)

# ---- arguments and outputs --------------------------------------------------------------------------------
args <- c("outputs/cell_alerts_incidence_M_fullsrc.csv.gz", NE_ZIP, file.path(FO, "fig4_cell_maps")) # was: args <- commandArgs(trailingOnly = TRUE); persistence input dropped 2026-11-05
if (length(args) != 3) stop("usage: Rscript fig_cell_maps.R <incidence_csv_gz> <ne_zip> <out_dir>")
f_inc <- args[1]; f_ne <- args[2]; out_dir <- args[3]
for (f in c(f_inc, f_ne)) if (!file.exists(f)) stop("input not found: ", f)
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
f_png <- file.path(out_dir, "fig_cell_alerts_maps_nested_M_fullsrc.png")
f_pdf <- file.path(out_dir, "fig_cell_alerts_maps_nested_M_fullsrc.pdf")
for (f in c(f_png, f_pdf)) if (file.exists(f)) stop("refusing to overwrite existing output: ", f)

# ---- cells ------------------------------------------------------------------------------------------------
bin_levels <- c("-2 or less", "-2 to -0.5", "-0.5 to 0", "No change", "0 to +0.5", "+0.5 to +2", "+2 or more")
inc <- read_csv(f_inc, show_col_types = FALSE)
need_inc <- c("gid", "region", "events", "tp_base", "fp_base", "tp_emb", "fp_emb", "n_seeds", "d_tp", "d_fp", "lat", "lon")
if (!all(need_inc %in% names(inc))) stop("incidence file missing columns: ", paste(setdiff(need_inc, names(inc)), collapse = ", "))
stopifnot(!anyDuplicated(inc$gid))
stopifnot(!anyNA(inc[need_inc]), all(inc$n_seeds == 3))
stopifnot(all(abs(inc$lat * 4 - round(inc$lat * 4)) < 1e-9), all(abs(inc$lon * 4 - round(inc$lon * 4)) < 1e-9)) # 0.5-deg centres
# d_tp / d_fp are the seed means of the per-seed differences: check against the stored seed-mean tp/fp columns
stopifnot(max(abs(inc$d_tp - (inc$tp_emb - inc$tp_base))) < 1e-3, max(abs(inc$d_fp - (inc$fp_emb - inc$fp_base))) < 1e-3)

bin_net <- function(d) {
  d |> mutate(net = d_tp - d_fp,
              bin = factor(case_when(net <= -2 ~ "-2 or less", net <= -0.5 ~ "-2 to -0.5", net < 0 ~ "-0.5 to 0",
                                     net == 0 ~ "No change", net < 0.5 ~ "0 to +0.5", net < 2 ~ "+0.5 to +2",
                                     TRUE ~ "+2 or more"), levels = bin_levels))
}
inc <- bin_net(inc)
stopifnot(!anyNA(inc$bin))

# ---- totals check against the text (Sec. 4.1) -------------------------------------------------------------
totals <- tibble(
  panel = c("A", "B"),
  target = c(710, 197),
  sum_d_tp = c(sum(inc$d_tp), sum(inc$d_tp[inc$region == "africa"])),
  sum_d_fp = c(sum(inc$d_fp), sum(inc$d_fp[inc$region == "africa"])),
  sum_net = c(sum(inc$net), sum(inc$net[inc$region == "africa"])),
  # Africa by map extent instead of held-out region label (for the error report only)
  sum_d_tp_extent = c(NA, sum(inc$d_tp[between(inc$lon, -25, 60) & between(inc$lat, -36, 38)]))
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
} else cat("[fig] totals reproduce Sec. 4.1 (extra correct alerts = sum(net)/2): A 710, B 197\n")

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
act_inc <- inc |> filter(events > 0 | net != 0)
cat(sprintf("[fig] cells drawn: nested %d (of %d)\n", nrow(act_inc), nrow(inc)))

pA <- ggplot() +
  geom_sf(data = world, fill = LAND, colour = BORDER, linewidth = 0.08) +
  geom_tile(data = act_inc, aes(lon, lat, fill = bin), width = 0.5, height = 0.5) +
  geom_sf(data = regions, fill = NA, colour = REGION_LINE, linewidth = 0.25) +
  fill_bins + coord_sf(xlim = GLB_X, ylim = GLB_Y, expand = FALSE, datum = NA) + labs(x = NULL, y = NULL) +
  theme_void(base_size = 12) + theme(legend.position = "none", plot.margin = margin(0, 0, 0, 0))
pB <- ggplot() +
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

# ---- layout (inches): one row, global map left, Africa right; titles above, legend at bottom ------
W <- 13; GAP <- 0.2
RA <- diff(GLB_X) / diff(GLB_Y); RB <- diff(AFR_X) / diff(AFR_Y)
HT <- (W - GAP) / (RA + RB); WA <- HT * RA; WB <- HT * RB
TITLE <- 0.45; LEG <- 1.15
yBot <- LEG                                   # bottom of A and B
H <- yBot + HT + TITLE
stopifnot(WA + GAP + WB <= W + 1e-9)
cat(sprintf("[fig] page %.2f x %.2f in, map height %.2f in\n", W, H, HT))
inch <- function(v) grid::unit(v, "in")
vp <- function(x, y, w, h) grid::viewport(x = inch(x), y = inch(y), width = inch(w), height = inch(h), just = c("left", "bottom"))
tgp <- grid::gpar(fontsize = 19, fontface = "bold")

for (dev in c("png", "pdf")) {
  if (dev == "png") png(f_png, width = W, height = H, units = "in", res = 300, bg = "white")
  if (dev == "pdf") cairo_pdf(f_pdf, width = W, height = H, bg = "white")
  grid::grid.newpage()
print(pA, vp = vp(0, yBot, WA, HT)); print(pB, vp = vp(WA + GAP, yBot, WB, HT))
  grid::grid.text("A. Base + embedding vs. base, global", x = inch(0.03), y = inch(H - 0.05), just = c("left", "top"), gp = tgp)
  grid::grid.text("B. Africa", x = inch(WA + GAP), y = inch(H - 0.05), just = c("left", "top"), gp = tgp)
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

# ===================== fig19: figures_tables/fig_cell_alerts.R (original body; single-figure mode only) =====================
if (ONE == "fig19") {
  cat("[figures.R] fig19 <- fig_cell_alerts.R (original body)\n")
  local({
# fig_cell_alerts.R -- Figure: where the pretrained embedding changes forecasts (ggplot2 only).
# Two columns (0.6 / 0.4): A = global PRIO-GRID map, B = Africa, same measure, cells and color scale.
# Measure per 0.5-degree cell: correct alerts gained minus false alerts added when the embedding is added to the
# base model, at equal alert budgets (cell_alerts_agg.py, Mode B forecasts, seed-averaged).
# Usage:  Rscript fig_cell_alerts.R <cell_alerts_{target}.csv.gz> <ne_50m_admin_0_countries.zip> <target> <out.png>
#   target = incidence | civilian
set.seed(123)
suppressPackageStartupMessages({
  library(ggplot2); library(sf); library(dplyr); library(readr); library(patchwork); library(scales)
})
args <- PASS   # was: args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 4)
f_cells <- args[1]; f_ne <- args[2]; target <- args[3]; f_out <- args[4]
stopifnot(file.exists(f_cells), file.exists(f_ne), target %in% c("incidence", "civilian"))
outcome <- c(incidence = "all political violence", civilian = "violence against civilians")[[target]]
sf_use_s2(FALSE)

# ---- cells ----------------------------------------------------------------------------------------------
cells <- read_csv(f_cells, show_col_types = FALSE)
stopifnot(all(c("gid", "region", "d_tp", "d_fp", "lat", "lon") %in% names(cells)), !anyDuplicated(cells$gid))
cells <- cells |> mutate(net = d_tp - d_fp)
cells <- cells |> mutate(bin = factor(case_when(net <= -2 ~ "-2 or less", net <= -0.5 ~ "-2 to -0.5", net < 0 ~ "-0.5 to 0",
                                                net == 0 ~ "No change", net < 0.5 ~ "0 to +0.5", net < 2 ~ "+0.5 to +2",
                                                TRUE ~ "+2 or more"),
                                      levels = c("-2 or less", "-2 to -0.5", "-0.5 to 0", "No change", "0 to +0.5",
                                                 "+0.5 to +2", "+2 or more")))
stopifnot(!anyNA(cells$bin))
shown <- cells |> filter(abs(net) >= 0.5)       # continuous version: at least half an alert (seed-averaged)
active <- cells |> filter(events > 0 | net != 0)   # binned version: every cell with events or any change
# region-level version of the same quantity: summed over cells, scaled per 100 event cell-months so large and small
# regions are comparable (Africa has ~3.7x the event cell-months of Asia-Pacific)
reg_rate <- cells |> group_by(region) |> summarise(net = sum(net), events = sum(events), n_seeds = min(n_seeds), .groups = "drop") |>
  mutate(rate = 100 * net / events)
stopifnot(nrow(reg_rate) == 6, all(reg_rate$events > 0))
print(reg_rate)
cat(sprintf("[fig] %s: %d cells, %d shown (|net| >= 0.5)\n", target, nrow(cells), nrow(shown)))

# ---- countries and held-out regions (hard-coded from scripts/phase4_scaling/regions.py, 2026-10-13) --------
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
# Natural Earth names that differ from the project's country names
alias <- c("Dem. Rep. Congo" = "Democratic Republic of Congo", "Congo" = "Republic of Congo",
           "Central African Rep." = "Central African Republic", "Côte d'Ivoire" = "Ivory Coast", "Eq. Guinea" = "Equatorial Guinea",
           "S. Sudan" = "South Sudan", "W. Sahara" = "Western Sahara", "Bosnia and Herz." = "Bosnia-Herzegovina",
           "Czechia" = "Czech Republic", "Macedonia" = "North Macedonia", "Dominican Rep." = "Dominican Republic",
           "Falkland Is." = "Falkland Islands", "Solomon Is." = "Solomon Islands", "N. Cyprus" = "Cyprus",
           "Somaliland" = "Somalia", "Kosovo" = "Serbia", "Hong Kong" = "China", "Macao" = "China", "Cabo Verde" = "Cape Verde")
tmp <- tempfile(); unzip(f_ne, exdir = tmp)
world <- st_read(list.files(tmp, pattern = "\\.shp$", full.names = TRUE)[1], quiet = TRUE) |>
  select(NAME) |> filter(NAME != "Antarctica") |>
  mutate(proj_name = ifelse(NAME %in% names(alias), alias[NAME], NAME)) |>
  left_join(lookup, by = "proj_name")
stopifnot(sum(!is.na(world$region_key)) >= 150)
# ---- country of each cell: centroid in polygon; coastal centroids that fall in the sea go to the nearest country ----
st_crs(world) <- 4326
pts <- st_as_sf(cells, coords = c("lon", "lat"), crs = 4326, remove = FALSE)
hit <- st_intersects(pts, st_make_valid(world))
ci <- vapply(hit, function(i) if (length(i)) i[1] else NA_integer_, integer(1))
cat(sprintf("[fig] %d of %d cell centroids off-land, assigned to nearest country\n", sum(is.na(ci)), length(ci)))
ci[is.na(ci)] <- st_nearest_feature(pts[is.na(ci), ], world)
cells$country <- world$NAME[ci]
MIN_EV <- 100   # countries with fewer event cell-months in the test period are shown as land only
ctry_rate <- cells |> group_by(country) |> summarise(net = sum(net), events = sum(events), .groups = "drop") |>
  mutate(rate = 100 * net / events)
ctry_bins <- c("-3 or less", "-3 to -1", "-1 to 0", "No change", "0 to +1", "+1 to +3", "+3 or more")
ctry_rate <- ctry_rate |> filter(events >= MIN_EV) |>
  mutate(bin = factor(case_when(rate <= -3 ~ "-3 or less", rate <= -1 ~ "-3 to -1", rate < 0 ~ "-1 to 0", rate == 0 ~ "No change",
                                rate < 1 ~ "0 to +1", rate < 3 ~ "+1 to +3", TRUE ~ "+3 or more"), levels = ctry_bins))
stopifnot(!anyNA(ctry_rate$bin))
cat(sprintf("[fig] countries with >= %d event cell-months: %d (gain %d, loss %d, zero %d)\n", MIN_EV, nrow(ctry_rate),
            sum(ctry_rate$rate > 0), sum(ctry_rate$rate < 0), sum(ctry_rate$rate == 0)))
regions <- world |> filter(!is.na(region_key)) |> st_make_valid() |> group_by(region_key) |> summarise(.groups = "drop")

# ---- shared pieces --------------------------------------------------------------------------------------
LAND <- "#3A3A3A"; BORDER <- "#4A4A4A"; REGION_LINE <- "#D0D0D0"
LIM <- 3
leg_title <- "Correct alerts gained minus false alerts added\nper 0.5\u00b0 cell (count of cell-months, test period)"
# continuous count scale: saturation and brightness grow with |change| so large changes stand out on dark land
fill_cont <- scale_fill_gradientn(colours = c("#FF3B30", "#C8463F", "#6E6E6E", "#3F7BC8", "#40A6FF"),
                                  values = rescale(c(-LIM, -1, 0, 1, LIM)), limits = c(-LIM, LIM), oob = squish,
                                  breaks = -3:3, labels = c("\u2264 -3", "-2", "-1", "0", "+1", "+2", "\u2265 +3"),
                                  name = leg_title)
# binned version of the same count: three loss and three gain classes, symmetric around zero; cells where the
# embedding changed nothing (exactly 0 after seed averaging) get their own muted class so they are not counted as gains
bin_levels <- c("-2 or less", "-2 to -0.5", "-0.5 to 0", "No change", "0 to +0.5", "+0.5 to +2", "+2 or more")
fill_bins <- scale_fill_manual(values = setNames(c("#FF3B30", "#E8746C", "#B9908C", "#5E5E5E", "#8FA9C9", "#5E9BE6", "#40A6FF"),
                                                 bin_levels), drop = FALSE, name = leg_title)
SB_LW <- 1.6; SB_SIZE <- 5; RL_LW <- 0.3   # screen defaults; print versions reset these
scale_bar <- function(x0, y0, km, tick) {
  # plate carree: 1 degree of longitude = 111.32 km * cos(latitude)
  dx <- km / (111.32 * cos(y0 * pi / 180))
  list(annotate("segment", x = x0, xend = x0 + dx, y = y0, yend = y0, linewidth = SB_LW, colour = "#111111"),
       annotate("segment", x = c(x0, x0 + dx), xend = c(x0, x0 + dx), y = y0 - tick, yend = y0 + tick,
                linewidth = 0.75 * SB_LW, colour = "#111111"),
       annotate("text", x = x0 + dx / 2, y = y0 + 1.8 * tick, label = paste0(format(km, big.mark = ","), " km"),
                size = SB_SIZE, fontface = "bold", colour = "#111111", vjust = 0))
}
map_panel <- function(xlim, ylim, bar, kind) {
  layer <- if (kind == "count") geom_tile(data = shown, aes(lon, lat, fill = net), width = 0.5, height = 0.5)
           else geom_tile(data = active, aes(lon, lat, fill = bin), width = 0.5, height = 0.5)
  ggplot() +
    geom_sf(data = world, fill = LAND, colour = BORDER, linewidth = 0.08) +
    layer +
    geom_sf(data = regions, fill = NA, colour = REGION_LINE, linewidth = RL_LW) +
    scale_bar(bar[1], bar[2], bar[3], bar[4]) +
    (if (kind == "count") fill_cont else fill_bins) +
    (if (kind == "count") guides(fill = guide_colourbar(title.position = "top", title.hjust = 0.5))
     else guides(fill = guide_legend(title.position = "top", title.hjust = 0.5, nrow = 1, label.position = "bottom"))) +
    coord_sf(xlim = xlim, ylim = ylim, expand = FALSE, datum = NA) +
    labs(x = NULL, y = NULL) +
    theme_void(base_size = 12) + theme(legend.position = "none", plot.margin = margin(0, 0, 0, 0))
}
# label anchors (lon, lat), hard-coded inside each region's largest landmass
lab_xy <- tibble(region = c("africa", "americas", "asia_pacific", "europe", "mena", "south_asia"),
                 lx = c(20, -58, 108, 45, 44, 78), ly = c(3, -10, 33, 58, 24, 22))
region_panel <- function(xlim, ylim) {
  rd <- regions |> left_join(reg_rate, by = c("region_key" = "region")) |> left_join(lab_xy, by = c("region_key" = "region"))
  stopifnot(!anyNA(rd$rate))
  rd <- rd |> mutate(lab = sprintf("%+.1f%s", rate, ifelse(n_seeds < 3, "*", "")))
  ggplot() +
    geom_sf(data = world, fill = LAND, colour = BORDER, linewidth = 0.08) +
    geom_sf(data = rd, aes(fill = rate), colour = REGION_LINE, linewidth = 0.3) +
    geom_label(data = st_drop_geometry(rd), aes(lx, ly, label = lab), size = 5.5, fontface = "bold",
               fill = "white", colour = "#111111", label.size = 0, label.padding = grid::unit(0.15, "lines")) +
    scale_fill_gradient(low = "#24425F", high = "#40A6FF", limits = c(0, 6), oob = squish, breaks = 0:6,
                        name = "Net correct alerts per 100 event cell-months, held-out region") +
    guides(fill = guide_colourbar(title.position = "top", title.hjust = 0.5)) +
    coord_sf(xlim = xlim, ylim = ylim, expand = FALSE, datum = NA) +
    labs(x = NULL, y = NULL) +
    theme_void(base_size = 12) + theme(legend.position = "none", plot.margin = margin(0, 0, 0, 0))
}
country_panel <- function(xlim, ylim, bar = NULL) {
  cd <- world |> inner_join(ctry_rate, by = c("NAME" = "country"))
  stopifnot(nrow(cd) == nrow(ctry_rate))
  ggplot() +
    geom_sf(data = world, fill = LAND, colour = BORDER, linewidth = 0.08) +
    geom_sf(data = cd, aes(fill = bin), colour = "#1E1E1E", linewidth = 0.15) +
    geom_sf(data = regions, fill = NA, colour = REGION_LINE, linewidth = 0.3) +
    (if (!is.null(bar)) scale_bar(bar[1], bar[2], bar[3], bar[4])) +
    scale_fill_manual(values = setNames(c("#FF3B30", "#E8746C", "#B9908C", "#5E5E5E", "#8FA9C9", "#5E9BE6", "#40A6FF"), ctry_bins),
                      drop = FALSE, name = sprintf("Net correct alerts per 100 event cell-months, country (\u2265 %d event cell-months)", MIN_EV)) +
    guides(fill = guide_legend(title.position = "top", title.hjust = 0.5, nrow = 1, label.position = "bottom")) +
    coord_sf(xlim = xlim, ylim = ylim, expand = FALSE, datum = NA) +
    labs(x = NULL, y = NULL) +
    theme_void(base_size = 12) + theme(legend.position = "none", plot.margin = margin(0, 0, 0, 0))
}
AFR_X <- c(-25, 60); AFR_Y <- c(-36, 38)
GLB_X <- c(-130, 180); GLB_Y <- c(-50, 75)   # cropped (no Alaska/Antarctic margin) so the global map renders larger
caption <- paste0("Mode B forecasts (made from January 2022 on), ", outcome, ". The forecast target is binary (any event in the ",
                  "cell-month); the mapped quantity counts cell-months over the test period.\nEqual alert budgets: base and ",
                  "base + embedding each flag as many cell-months as had events. Net = extra correct alerts minus extra false alarms ",
                  "(base + embedding vs. base); blue = positive net. Within a region the two cancel, so a region's net is twice its extra correct alerts; a blue cell or country can still have extra false alarms. Seed-averaged.\nRegion and country panels sum the same quantity per 100 event cell-months (countries need \u2265 100 event cell-months). * = MENA interim (1 seed).")
draw_figure <- function(kind, f_png, a_level = "region", c_level = "cells") {
  # Layout (inches). Left column (0.55): A (region or country totals) with its legend below, then B = 0.5-degree cells
  # with its legend below. Right column (0.45): C = Africa (cells or countries), bottom-aligned with B so the B and C
  # scale bars share one page height.
  W <- 16; WA <- 0.55 * W; WB <- 0.45 * W; GAP <- 0.12
  HA <- (WA - GAP) * diff(GLB_Y) / diff(GLB_X); HB <- WB * diff(AFR_Y) / diff(AFR_X)
  TITLE <- 0.55; CAP <- 0.6; LEG <- 1.3; LEGA <- 1.15
  yB0 <- CAP + LEG + 0.1
  yLA <- yB0 + HA + TITLE                             # bottom of legend A
  yA0 <- yLA + LEGA                                   # bottom of A
  H <- yA0 + HA + TITLE
  yC0 <- yB0; stopifnot(yC0 + HB + TITLE <= H)
  bar_y <- yB0 + 0.45
  lat_at <- function(y, y0, ylim, h) ylim[1] + (y - y0) * diff(ylim) / h
  tick_deg <- function(ylim, h) 0.07 * diff(ylim) / h
  pA <- if (a_level == "region") region_panel(GLB_X, GLB_Y) else country_panel(GLB_X, GLB_Y)
  pB <- map_panel(GLB_X, GLB_Y, c(-122, lat_at(bar_y, yB0, GLB_Y, HA), 2000, tick_deg(GLB_Y, HA)), kind)
  barC <- c(-23, lat_at(bar_y, yC0, AFR_Y, HB), 1000, tick_deg(AFR_Y, HB))
  pC <- if (c_level == "cells") map_panel(AFR_X, AFR_Y, barC, kind) else country_panel(AFR_X, AFR_Y, barC)
  png(f_png, width = W, height = H, units = "in", res = 300, bg = "white")
  grid::grid.newpage()
  inch <- function(v) grid::unit(v, "in")
  vp <- function(x, y, w, h) grid::viewport(x = inch(x), y = inch(y), width = inch(w), height = inch(h), just = c("left", "bottom"))
  print(pA, vp = vp(0, yA0, WA - GAP, HA))
  print(pB, vp = vp(0, yB0, WA - GAP, HA))
  print(pC, vp = vp(WA, yC0, WB, HB))
  tgp <- grid::gpar(fontsize = 20, fontface = "bold")
  ttlA <- if (a_level == "region") "A. Held-out regions" else "A. Countries"
  ttlC <- if (c_level == "cells") "C. 0.5\u00b0 cells, Africa" else "C. Countries, Africa"
  grid::grid.text(ttlA, x = inch(0.05), y = inch(H - 0.1), just = c("left", "top"), gp = tgp)
  grid::grid.text("B. 0.5\u00b0 cells, global", x = inch(0.05), y = inch(yB0 + HA + TITLE - 0.1), just = c("left", "top"), gp = tgp)
  grid::grid.text(ttlC, x = inch(WA), y = inch(yC0 + HB + TITLE - 0.1), just = c("left", "top"), gp = tgp)
  grab_leg <- function(p, kw) {
    g <- ggplotGrob(p + theme(legend.position = "bottom", legend.title = element_text(size = 13, face = "bold"),
                              legend.text = element_text(size = 12), legend.key.width = grid::unit(kw, "cm"),
                              legend.key.height = grid::unit(0.45, "cm"), legend.spacing.x = grid::unit(0, "cm")))
    g$grobs[[which(sapply(g$grobs, function(x) x$name) == "guide-box")]]
  }
  grid::pushViewport(vp(0, yLA, WA - GAP, LEGA)); grid::grid.draw(grab_leg(pA, if (a_level == "region") 1.6 else 2.2)); grid::popViewport()
  grid::pushViewport(vp(0, CAP, WA - GAP, LEG)); grid::grid.draw(grab_leg(pB, if (kind == "count") 3.2 else 2.2)); grid::popViewport()
  if (c_level == "country" && a_level == "region") {   # C uses the country scale; put its legend under C
    grid::pushViewport(vp(WA, CAP, WB, LEG)); grid::grid.draw(grab_leg(pC, 2.25)); grid::popViewport()
  }
  grid::grid.text(caption, x = inch(0.1), y = inch(0.08), just = c("left", "bottom"), gp = grid::gpar(fontsize = 9.5, col = "#555555"))
  dev.off()
  cat(sprintf("[fig] wrote %s\n", f_png))
}
REG_LAB <- c(africa = "Africa", americas = "Americas", asia_pacific = "Asia-Pacific", europe = "Europe", mena = "MENA",
             south_asia = "South Asia")
draw_2x2 <- function(f_png) {
  # Top row: A = global cells, B = Africa cells (same row height, so scale bars share a baseline).
  # Bottom row: C = countries, sorted dot plot by region; D = held-out region totals.
  W <- 16; GAP <- 0.15
  RA <- diff(GLB_X) / diff(GLB_Y); RB <- diff(AFR_X) / diff(AFR_Y)
  HT <- (W - GAP) / (RA + RB); WA <- HT * RA; WB <- HT * RB          # equal heights, widths from map aspect
  TITLE <- 0.55; CAP <- 0.75; LEG <- 1.25; HBOT <- 11; LEGC <- 2.1
  yBot <- CAP + LEGC                                   # bottom of C and D
  yLeg <- yBot + HBOT + TITLE                          # bottom of map legend
  yTop <- yLeg + LEG                                   # bottom of A and B
  H <- yTop + HT + TITLE
  bar_y <- yTop + 0.4
  lat_at <- function(y, y0, ylim, h) ylim[1] + (y - y0) * diff(ylim) / h
  tick_deg <- function(ylim, h) 0.07 * diff(ylim) / h
  pA <- map_panel(GLB_X, GLB_Y, c(-122, lat_at(bar_y, yTop, GLB_Y, HT), 2000, tick_deg(GLB_Y, HT)), "bins")
  pB <- map_panel(AFR_X, AFR_Y, c(-23, lat_at(bar_y, yTop, AFR_Y, HT), 1000, tick_deg(AFR_Y, HT)), "bins")

  # C: one dot per country (>= MIN_EV event cell-months), x clipped to +-10 with the clipped values printed
  XL <- 10
  cd <- cells |> group_by(country) |> summarise(region = names(which.max(table(region))), .groups = "drop") |>
    inner_join(ctry_rate, by = "country") |>
    mutate(region_lab = factor(REG_LAB[region], levels = REG_LAB), x = pmax(pmin(rate, XL), -XL),
           country = recode(country, "Dem. Rep. Congo" = "DR Congo", "Central African Rep." = "CAR", "S. Sudan" = "South Sudan",
                            "United States of America" = "United States", "C\u00f4te d'Ivoire" = "Ivory Coast"))
  stopifnot(nrow(cd) == nrow(ctry_rate), !anyNA(cd$region_lab))
  cd <- cd |> arrange(region_lab, rate) |> mutate(country = factor(country, levels = unique(country)))
  clipped <- cd |> filter(abs(rate) > XL)
  pC <- ggplot(cd, aes(x, country)) +
    geom_vline(xintercept = 0, colour = "#888888", linewidth = 0.4) +
    geom_segment(aes(x = 0, xend = x, yend = country), colour = "#BBBBBB", linewidth = 0.4) +
    geom_point(aes(fill = bin, size = events), shape = 21, colour = "#222222", stroke = 0.3) +
    geom_text(data = clipped, aes(label = sprintf("%+.1f", rate), hjust = ifelse(rate > 0, 1.5, -0.5), vjust = 0.35), size = 3.8, fontface = "bold") +
    scale_fill_manual(values = setNames(c("#FF3B30", "#E8746C", "#B9908C", "#5E5E5E", "#8FA9C9", "#5E9BE6", "#40A6FF"), ctry_bins),
                      drop = FALSE, guide = "none") +
    scale_size_area(max_size = 6, breaks = c(500, 2000, 5000), labels = comma, name = "Event cell-months") +
    scale_x_continuous(limits = c(-XL, XL), breaks = seq(-10, 10, 5), labels = function(v) sprintf("%+d", v)) +
    facet_grid(region_lab ~ ., scales = "free_y", space = "free_y",
               labeller = as_labeller(function(v) sub("Asia-Pacific", "Asia-\nPacific", v))) +
    labs(x = "Net correct alerts per 100 event cell-months", y = NULL) +
    theme_minimal(base_size = 11) +
    theme(panel.grid.major.y = element_blank(), panel.grid.minor = element_blank(), axis.text.y = element_text(size = 11.5, colour = "#111111"),
          strip.text.y = element_text(angle = -90, face = "bold", size = 12.5, lineheight = 0.85), strip.background = element_rect(fill = "#EEEEEE", colour = NA), legend.position = "bottom",
          panel.spacing.y = grid::unit(0.25, "lines"), axis.title.x = element_text(size = 12))

  # D: held-out region totals, one bar per region facet, rows aligned with C (same facets, free space by country count)
  nC <- cd |> count(region_lab)
  rd <- reg_rate |> mutate(region_lab = factor(REG_LAB[region], levels = REG_LAB)) |> left_join(nC, by = "region_lab") |>
    mutate(mid = (n + 1) / 2)
  stopifnot(!anyNA(rd$n))
  BAR <- 1.4   # bar thickness in country rows
  pD <- ggplot(rd) +
    geom_blank(aes(y = 0.5)) + geom_blank(aes(y = n + 0.5)) +
    geom_rect(aes(xmin = 0, xmax = rate, ymin = mid - BAR / 2, ymax = mid + BAR / 2), fill = "#1F1F1F") +   # neutral: not in the A-C scale
    geom_text(aes(x = rate, y = mid, label = sprintf("%+.1f%s", rate, ifelse(n_seeds < 3, "*", ""))),
              hjust = -0.2, size = 5, fontface = "bold") +
    geom_vline(xintercept = 0, colour = "#888888", linewidth = 0.4) +
    facet_grid(region_lab ~ ., scales = "free_y", space = "free_y") +
    scale_x_continuous(limits = c(0, 7), breaks = 0:6, expand = c(0, 0)) +
    scale_y_continuous(expand = c(0, 0)) +
    labs(x = "Net correct alerts per 100 event cell-months", y = NULL) +
    theme_minimal(base_size = 11) +
    theme(panel.grid.major.y = element_blank(), panel.grid.minor = element_blank(), axis.text.y = element_blank(),
          strip.text.y = element_blank(), panel.spacing.y = grid::unit(0.25, "lines"), axis.title.x = element_text(size = 12),
          plot.margin = margin(5, 15, 5, 5))
  pC <- pC + scale_y_discrete(expand = c(0, 0.5))
  png(f_png, width = W, height = H, units = "in", res = 300, bg = "white")
  grid::grid.newpage()
  inch <- function(v) grid::unit(v, "in")
  vp <- function(x, y, w, h) grid::viewport(x = inch(x), y = inch(y), width = inch(w), height = inch(h), just = c("left", "bottom"))
  print(pA, vp = vp(0, yTop, WA, HT))
  print(pB, vp = vp(WA + GAP, yTop, WB, HT))
  bottom <- (pC + theme(legend.position = "none")) + pD + plot_layout(widths = c(WA, WB))
  print(bottom, vp = vp(0, yBot, W, HBOT))
  tgp <- grid::gpar(fontsize = 20, fontface = "bold")
  grid::grid.text("A. 0.5\u00b0 cells, global", x = inch(0.05), y = inch(H - 0.1), just = c("left", "top"), gp = tgp)
  grid::grid.text("B. 0.5\u00b0 cells, Africa", x = inch(WA + GAP), y = inch(H - 0.1), just = c("left", "top"), gp = tgp)
  grid::grid.text("C. Countries", x = inch(0.05), y = inch(yBot + HBOT + TITLE - 0.1), just = c("left", "top"), gp = tgp)
  grid::grid.text("D. Held-out regions", x = inch(WA + GAP), y = inch(yBot + HBOT + TITLE - 0.1), just = c("left", "top"), gp = tgp)
  grab_leg <- function(g) g$grobs[[which(sapply(g$grobs, function(x) x$name) == "guide-box")]]
  legAB <- grab_leg(ggplotGrob(pA + theme(legend.position = "bottom", legend.title = element_text(size = 13, face = "bold"),
                                          legend.text = element_text(size = 12), legend.key.width = grid::unit(2.4, "cm"),
                                          legend.key.height = grid::unit(0.45, "cm"))))
  grid::pushViewport(vp(0, yLeg, W, LEG)); grid::grid.draw(legAB); grid::popViewport()
  # C legend: same seven colours with country breaks, plus dot size
  legC <- grab_leg(ggplotGrob(pC + guides(fill = guide_legend(title = "Country class", nrow = 1, title.position = "top",
                                                              override.aes = list(size = 7), order = 1),
                                          size = guide_legend(title = "Event cell-months", title.position = "top", nrow = 1, order = 2)) +
                                theme(legend.box = "vertical", legend.box.just = "left", legend.title = element_text(size = 15, face = "bold"),
                                      legend.text = element_text(size = 14), legend.spacing.y = grid::unit(0.3, "cm"))))
  grid::pushViewport(vp(0, CAP, WA, LEGC)); grid::grid.draw(legC); grid::popViewport()
  cap <- paste0(sub("\nRegion and country panels.*$", "", caption),
                "\nC and D: same quantity summed per 100 event cell-months; countries need \u2265 ", MIN_EV,
                " event cell-months; values beyond \u00b1", XL, " printed. D rows align with C. D bars are seed means (per-seed dots pending). * = MENA interim (1 seed).")
  grid::grid.text(cap, x = inch(0.1), y = inch(0.08), just = c("left", "bottom"), gp = grid::gpar(fontsize = 9.5, col = "#555555"))
  dev.off()
  cat(sprintf("[fig] wrote %s\n", f_png))
}

# ---- print versions (true size for a 6.5 in text block; fonts in points as printed; caption goes in LaTeX) --------
PT <- 1 / ggplot2::.pt          # geom text size per printed point
country_dots <- function(min_ev, name_pt) {
  cr <- cells |> group_by(country) |> summarise(region = names(which.max(table(region))), net = sum(net), events = sum(events),
                                                .groups = "drop") |>
    filter(events >= min_ev) |> mutate(rate = 100 * net / events)
  cr <- cr |> mutate(bin = factor(case_when(rate <= -3 ~ "-3 or less", rate <= -1 ~ "-3 to -1", rate < 0 ~ "-1 to 0", rate == 0 ~ "No change",
                                            rate < 1 ~ "0 to +1", rate < 3 ~ "+1 to +3", TRUE ~ "+3 or more"), levels = ctry_bins),
                     region_lab = factor(REG_LAB[region], levels = REG_LAB), x = pmax(pmin(rate, 10), -10),
                     country = recode(country, "Dem. Rep. Congo" = "DR Congo", "Central African Rep." = "CAR", "S. Sudan" = "South Sudan",
                                      "United States of America" = "United States", "C\u00f4te d'Ivoire" = "Ivory Coast",
                                      "Trinidad and Tobago" = "Trinidad & Tobago")) |>
    arrange(region_lab, rate) |> mutate(country = factor(country, levels = unique(country)))
  stopifnot(!anyNA(cr$bin), !anyNA(cr$region_lab))
  clipped <- cr |> filter(abs(rate) > 10)
  n_reg <- table(cr$region_lab)
  strip_lab <- function(v) {   # wrap long names; abbreviate a region whose block is under 3 rows so the label fits
    out <- sub("South Asia", "South\nAsia", sub("Asia-Pacific", "Asia-\nPacific", v))
    ifelse(n_reg[v] < 3, c(Africa = "Afr.", Americas = "Amer.", `Asia-Pacific` = "A-P", Europe = "Eur.", MENA = "MENA",
                            `South Asia` = "S. As.")[v], out)
  }
  p <- ggplot(cr, aes(x, country)) +
    geom_vline(xintercept = 0, colour = "#888888", linewidth = 0.3) +
    geom_segment(aes(x = 0, xend = x, yend = country), colour = "#BBBBBB", linewidth = 0.3) +
    geom_point(aes(fill = bin, size = events), shape = 21, colour = "#222222", stroke = 0.2) +
    geom_text(data = clipped, aes(label = sprintf("%+.1f", rate), hjust = ifelse(rate > 0, 1.4, -0.4)), vjust = 0.35,
              size = 6 * PT, fontface = "bold") +
    scale_fill_manual(values = setNames(c("#FF3B30", "#E8746C", "#B9908C", "#5E5E5E", "#8FA9C9", "#5E9BE6", "#40A6FF"), ctry_bins),
                      drop = FALSE, name = "Country class") +
    scale_size_area(max_size = 3.2, breaks = c(500, 2000, 5000), labels = comma, name = "Event cell-months") +
    scale_x_continuous(limits = c(-10, 10), breaks = seq(-10, 10, 5), labels = function(v) sprintf("%+d", v)) +
    scale_y_discrete(expand = c(0, 0.5)) +
    facet_grid(region_lab ~ ., scales = "free_y", space = "free_y",
               labeller = as_labeller(strip_lab)) +
    labs(x = "Net correct alerts per 100 event cell-months", y = NULL) +
    theme_minimal(base_size = 7) +
    theme(panel.grid.major.y = element_blank(), panel.grid.minor = element_blank(),
          axis.text.y = element_text(size = name_pt, colour = "#111111"), axis.text.x = element_text(size = 6.5),
          strip.text.y = element_text(angle = -90, face = "bold", size = 6.2, lineheight = 0.8, margin = margin(1, 1, 1, 1)),
          strip.background = element_rect(fill = "#EEEEEE", colour = NA), panel.spacing.y = grid::unit(0.15, "lines"),
          axis.title.x = element_text(size = 7.5), plot.margin = margin(2, 2, 2, 2))
  n <- cr |> count(region_lab)
  rd <- reg_rate |> mutate(region_lab = factor(REG_LAB[region], levels = REG_LAB)) |> left_join(n, by = "region_lab") |>
    mutate(mid = (n + 1) / 2)
  stopifnot(!anyNA(rd$n))
  BARW <- min(1.4, 0.9 * min(rd$n))
  d <- ggplot(rd) +
    geom_blank(aes(y = 0.5)) + geom_blank(aes(y = n + 0.5)) +
    geom_rect(aes(xmin = 0, xmax = rate, ymin = mid - BARW / 2, ymax = mid + BARW / 2), fill = "#1F1F1F") +
    geom_text(aes(x = rate, y = mid, label = sprintf("%+.1f%s", rate, ifelse(n_seeds < 3, "*", ""))), hjust = -0.2,
              size = 7 * PT, fontface = "bold") +
    geom_vline(xintercept = 0, colour = "#888888", linewidth = 0.3) +
    facet_grid(region_lab ~ ., scales = "free_y", space = "free_y") +
    scale_x_continuous(limits = c(0, 7), breaks = 0:6, expand = c(0, 0)) +
    scale_y_continuous(expand = c(0, 0)) +
    labs(x = "Per 100 event cell-months", y = NULL) +
    theme_minimal(base_size = 7) +
    theme(panel.grid.major.y = element_blank(), panel.grid.minor = element_blank(), axis.text.y = element_blank(),
          axis.text.x = element_text(size = 6.5), strip.text.y = element_blank(), panel.spacing.y = grid::unit(0.15, "lines"),
          axis.title.x = element_text(size = 7.5), plot.margin = margin(2, 6, 2, 4))
  list(C = p, D = d, n = nrow(cr))
}
draw_print <- function(f_png, W, H, parts, min_ev = 300, name_pt = 7, letters = c("A", "B", "C", "D")) {
  # parts: "maps", "dots" or c("maps", "dots"); all sizes are inches/points as printed
  SB_LW <<- 0.7; SB_SIZE <<- 6.5 * PT; RL_LW <<- 0.12
  GAP <- 0.08; TITLE <- 0.22; LEG <- 0.5; LEGC <- 0.62
  RA <- diff(GLB_X) / diff(GLB_Y); RB <- diff(AFR_X) / diff(AFR_Y)
  HT <- (W - GAP) / (RA + RB); WA <- HT * RA; WB <- HT * RB
  png(f_png, width = W, height = H, units = "in", res = 600, bg = "white")
  grid::grid.newpage()
  inch <- function(v) grid::unit(v, "in")
  vp <- function(x, y, w, h) grid::viewport(x = inch(x), y = inch(y), width = inch(w), height = inch(h), just = c("left", "bottom"))
  tgp <- grid::gpar(fontsize = 10, fontface = "bold")
  grab_leg <- function(g) g$grobs[[which(sapply(g$grobs, function(x) x$name) == "guide-box")]]
  y <- H; li <- 1
  if ("maps" %in% parts) {
    yTop <- y - TITLE - HT
    bar_y <- yTop + 0.17
    lat_at <- function(yy, y0, ylim, h) ylim[1] + (yy - y0) * diff(ylim) / h
    tick_deg <- function(ylim, h) 0.05 * diff(ylim) / h
    pA <- map_panel(GLB_X, GLB_Y, c(-122, lat_at(bar_y, yTop, GLB_Y, HT), 2000, tick_deg(GLB_Y, HT)), "bins")
    pB <- map_panel(AFR_X, AFR_Y, c(-20, lat_at(bar_y, yTop, AFR_Y, HT), 1000, tick_deg(AFR_Y, HT)), "bins")
    print(pA, vp = vp(0, yTop, WA, HT)); print(pB, vp = vp(WA + GAP, yTop, WB, HT))
    grid::grid.text(paste0(letters[li], ". 0.5\u00b0 cells, global"), x = inch(0.02), y = inch(y - 0.03), just = c("left", "top"), gp = tgp)
    grid::grid.text(paste0(letters[li + 1], ". 0.5\u00b0 cells, Africa"), x = inch(WA + GAP), y = inch(y - 0.03), just = c("left", "top"), gp = tgp)
    leg <- grab_leg(ggplotGrob(pA + theme(legend.position = "bottom", legend.title = element_text(size = 7.5, face = "bold"),
                                          legend.text = element_text(size = 7), legend.key.width = grid::unit(0.95, "cm"),
                                          legend.key.height = grid::unit(0.22, "cm"))))
    grid::pushViewport(vp(0, yTop - LEG, W, LEG)); grid::grid.draw(leg); grid::popViewport()
    y <- yTop - LEG; li <- li + 2
  }
  if ("dots" %in% parts) {
    cdp <- country_dots(min_ev, name_pt)
    yBot <- LEGC
    grid::grid.text(paste0(letters[li], ". Countries (\u2265 ", min_ev, " event cell-months)"), x = inch(0.02), y = inch(y - 0.03),
                    just = c("left", "top"), gp = tgp)
    grid::grid.text(paste0(letters[li + 1], ". Held-out regions"), x = inch(WA + GAP), y = inch(y - 0.03), just = c("left", "top"), gp = tgp)
    bottom <- (cdp$C + theme(legend.position = "none")) + cdp$D + plot_layout(widths = c(WA, WB))
    print(bottom, vp = vp(0, yBot, W, y - TITLE - yBot))
    legC <- grab_leg(ggplotGrob(cdp$C + guides(fill = guide_legend(nrow = 1, title.position = "left", override.aes = list(size = 2.6), order = 1),
                                               size = guide_legend(nrow = 1, title.position = "left", order = 2)) +
                                  theme(legend.position = "bottom", legend.box = "vertical", legend.box.just = "left",
                                        legend.title = element_text(size = 7.5, face = "bold"), legend.text = element_text(size = 7),
                                        legend.spacing.y = grid::unit(0.05, "cm"), legend.key.size = grid::unit(0.3, "cm"))))
    grid::pushViewport(vp(0, 0, W, LEGC)); grid::grid.draw(legC); grid::popViewport()
    cat(sprintf("[fig] %s: %d countries (>= %d event cell-months), name font %.1f pt\n", basename(f_png), cdp$n, min_ev, name_pt))
  }
  dev.off()
  SB_LW <<- 1.6; SB_SIZE <<- 5; RL_LW <<- 0.3
  cat(sprintf("[fig] wrote %s (%.2f x %.2f in)\n", f_png, W, H))
}
# Option 1: one full-page figure, 6.5 x 9 in, countries >= 300 event cell-months
draw_print(sub("\\.png$", "_print_opt1.png", f_out), W = 6.5, H = 9, parts = c("maps", "dots"), min_ev = 300, name_pt = 7)
# Option 4: two figures. Maps 6.5 in wide; countries (all >= 100) on their own full page
RA_ <- diff(GLB_X) / diff(GLB_Y); RB_ <- diff(AFR_X) / diff(AFR_Y)
draw_print(sub("\\.png$", "_print_opt4_maps.png", f_out), W = 6.5, H = 0.22 + (6.5 - 0.08) / (RA_ + RB_) + 0.5 + 0.02, parts = "maps")
draw_print(sub("\\.png$", "_print_opt4_countries.png", f_out), W = 6.5, H = 9, parts = "dots", min_ev = 100, name_pt = 6.8,
           letters = c("A", "B"))

# ---- option 4b: countries figure in landscape-friendly shape (two columns of regions; region total as dashed line) ----
draw_countries_wide <- function(f_png, W = 6.5, min_ev = 100, name_pt = 6.8) {
  cr <- cells |> group_by(country) |> summarise(region = names(which.max(table(region))), net = sum(net), events = sum(events),
                                                .groups = "drop") |>
    filter(events >= min_ev) |> mutate(rate = 100 * net / events) |>
    mutate(bin = factor(case_when(rate <= -3 ~ "-3 or less", rate <= -1 ~ "-3 to -1", rate < 0 ~ "-1 to 0", rate == 0 ~ "No change",
                                  rate < 1 ~ "0 to +1", rate < 3 ~ "+1 to +3", TRUE ~ "+3 or more"), levels = ctry_bins),
           region_lab = factor(REG_LAB[region], levels = REG_LAB), x = pmax(pmin(rate, 10), -10),
           country = recode(country, "Dem. Rep. Congo" = "DR Congo", "Central African Rep." = "CAR", "S. Sudan" = "South Sudan",
                            "United States of America" = "United States", "C\u00f4te d'Ivoire" = "Ivory Coast",
                            "Trinidad and Tobago" = "Trinidad & Tobago")) |>
    arrange(region_lab, rate) |> mutate(country = factor(country, levels = unique(country)))
  stopifnot(!anyNA(cr$bin), !anyNA(cr$region_lab))
  rr <- reg_rate |> mutate(region_lab = factor(REG_LAB[region], levels = REG_LAB),
                           lab = sprintf("Region %+.1f%s", rate, ifelse(n_seeds < 3, "*", "")))
  col_of <- c(Africa = 1, `Asia-Pacific` = 1, Europe = 1, Americas = 2, MENA = 2, `South Asia` = 2)
  one_col <- function(k, show_x) {
    d <- cr |> filter(col_of[as.character(region_lab)] == k) |> droplevels()
    r <- rr |> filter(region_lab %in% levels(d$region_lab)) |> mutate(region_lab = factor(as.character(region_lab), levels = levels(d$region_lab)))
    top <- d |> group_by(region_lab) |> summarise(ytop = levels(d$country)[min(as.integer(country))], .groups = "drop")
    r <- r |> left_join(top, by = "region_lab")
    clipped <- d |> filter(abs(rate) > 10)
    ggplot(d, aes(x, country)) +
      geom_vline(xintercept = 0, colour = "#888888", linewidth = 0.3) +
      geom_vline(data = r, aes(xintercept = rate), colour = "#1F1F1F", linewidth = 0.45, linetype = "22") +
      geom_segment(aes(x = 0, xend = x, yend = country), colour = "#BBBBBB", linewidth = 0.3) +
      geom_point(aes(fill = bin, size = events), shape = 21, colour = "#222222", stroke = 0.2) +
      geom_label(data = r, aes(x = rate, y = ytop, label = lab), inherit.aes = FALSE, hjust = -0.05, vjust = 0.5, size = 6 * PT,
                 fontface = "bold", fill = "white", label.size = 0, label.padding = grid::unit(0.08, "lines"), nudge_y = 0) +
      geom_text(data = clipped, aes(label = sprintf("%+.1f", rate), hjust = ifelse(rate > 0, 1.4, -0.4)), vjust = 0.35,
                size = 6 * PT, fontface = "bold") +
      scale_fill_manual(values = setNames(c("#FF3B30", "#E8746C", "#B9908C", "#5E5E5E", "#8FA9C9", "#5E9BE6", "#40A6FF"), ctry_bins),
                        drop = FALSE, name = "Country class") +
      scale_size_area(max_size = 3.2, breaks = c(500, 2000, 5000), labels = comma, name = "Event cell-months", limits = c(0, max(cr$events))) +
      scale_x_continuous(limits = c(-10, 10), breaks = seq(-10, 10, 5), labels = function(v) sprintf("%+d", v)) +
      scale_y_discrete(expand = c(0, 0.6)) +
      facet_grid(region_lab ~ ., scales = "free_y", space = "free_y",
                 labeller = as_labeller(function(v) sub("South Asia", "South\nAsia", sub("Asia-Pacific", "Asia-\nPacific", v)))) +
      labs(x = NULL, y = NULL) +
      theme_minimal(base_size = 7) +
      theme(panel.grid.major.y = element_blank(), panel.grid.minor = element_blank(),
            axis.text.y = element_text(size = name_pt, colour = "#111111"), axis.text.x = element_text(size = 6.5),
            strip.text.y = element_text(angle = -90, face = "bold", size = 6.5, lineheight = 0.8, margin = margin(1, 1, 1, 1)),
            strip.background = element_rect(fill = "#EEEEEE", colour = NA), panel.spacing.y = grid::unit(0.2, "lines"),
            axis.title.x = element_text(size = 7.5), plot.margin = margin(2, 4, 2, 2), legend.position = "none")
  }
  p1 <- one_col(1); p2 <- one_col(2)
  n1 <- sum(col_of[as.character(cr$region_lab)] == 1); n2 <- nrow(cr) - n1
  ROW <- 0.118; TITLE <- 0.0; LEGC <- 0.62; AX <- 0.25; XT <- 0.18
  H <- LEGC + XT + AX + max(n1, n2) * ROW + 0.35
  png(f_png, width = W, height = H, units = "in", res = 600, bg = "white")
  grid::grid.newpage()
  inch <- function(v) grid::unit(v, "in")
  vp <- function(x, y, w, h) grid::viewport(x = inch(x), y = inch(y), width = inch(w), height = inch(h), just = c("left", "bottom"))
  print(p1 + p2 + plot_layout(widths = c(1, 1)), vp = vp(0, LEGC + XT, W, H - LEGC - XT))
  grid::grid.text("Net correct alerts per 100 event cell-months", x = inch(W / 2), y = inch(LEGC + XT / 2),
                  gp = grid::gpar(fontsize = 7.5))   # one shared x-axis title, centred
  grab_leg <- function(g) g$grobs[[which(sapply(g$grobs, function(x) x$name) == "guide-box")]]
  legC <- grab_leg(ggplotGrob(p1 + guides(fill = guide_legend(nrow = 1, title.position = "left", override.aes = list(size = 2.6), order = 1),
                                          size = guide_legend(nrow = 1, title.position = "left", order = 2)) +
                                theme(legend.position = "bottom", legend.box = "vertical", legend.box.just = "left",
                                      legend.title = element_text(size = 7, face = "bold"), legend.text = element_text(size = 6.5, margin = margin(l = 1, r = 3)),
                                      legend.spacing.x = grid::unit(0.05, "cm"), legend.key.size = grid::unit(0.24, "cm"), legend.box.spacing = grid::unit(0, "cm"))))
  grid::pushViewport(vp(0, 0, W, LEGC)); grid::grid.draw(legC); grid::popViewport()
  dev.off()
  cat(sprintf("[fig] wrote %s (%.2f x %.2f in), %d countries\n", f_png, W, H, nrow(cr)))
}
draw_countries_wide(sub("\\.png$", "_print_opt4b_countries_wide.png", f_out))

# ---- all countries with any event (176), three columns of regions; measure = "rate" or "count" ----------------------
draw_countries_all <- function(f_png, measure = c("rate", "count"), W = 6.5, name_pt = 6, small_ev = 100) {
  measure <- match.arg(measure)
  cr <- cells |> group_by(country) |> summarise(region = names(which.max(table(region))), net = sum(net), events = sum(events),
                                                .groups = "drop") |>
    filter(events > 0) |> mutate(rate = 100 * net / events, small = events < small_ev)
  stopifnot(nrow(cr) == 176)
  if (measure == "rate") {
    bins <- ctry_bins; cr <- cr |> mutate(val = rate, x = pmax(pmin(rate, 10), -10))
    cr <- cr |> mutate(bin = case_when(val <= -3 ~ bins[1], val <= -1 ~ bins[2], val < 0 ~ bins[3], val == 0 ~ bins[4],
                                       val < 1 ~ bins[5], val < 3 ~ bins[6], TRUE ~ bins[7]))
    xlab <- "Net correct alerts per 100 event cell-months"
  } else {
    bins <- c("-10 or less", "-10 to -2", "-2 to 0", "No change", "0 to +2", "+2 to +10", "+10 or more")
    cr <- cr |> mutate(val = net, x = net)
    cr <- cr |> mutate(bin = case_when(val <= -10 ~ bins[1], val <= -2 ~ bins[2], val < 0 ~ bins[3], val == 0 ~ bins[4],
                                       val < 2 ~ bins[5], val < 10 ~ bins[6], TRUE ~ bins[7]))
    xlab <- "Net correct alerts (cell-months, test period), pseudo-log scale"
  }
  cr <- cr |> mutate(bin = factor(bin, levels = bins), region_lab = factor(REG_LAB[region], levels = REG_LAB),
                     country = recode(country, "Dem. Rep. Congo" = "DR Congo", "Central African Rep." = "CAR", "S. Sudan" = "South Sudan",
                                      "United States of America" = "United States", "C\u00f4te d'Ivoire" = "Ivory Coast",
                                      "Trinidad and Tobago" = "Trinidad & Tobago", "Bosnia and Herz." = "Bosnia & Herz.",
                                      "Dominican Rep." = "Dominican Rep.", "Eq. Guinea" = "Equatorial Guinea",
                                      "St. Vin. and Gren." = "St. Vincent & Gren.", "Solomon Is." = "Solomon Islands")) |>
    arrange(region_lab, val, events) |> mutate(country = factor(country, levels = unique(country)))
  stopifnot(!anyNA(cr$bin), !anyNA(cr$region_lab), !anyDuplicated(levels(cr$country)))
  rr <- reg_rate |> mutate(region_lab = factor(REG_LAB[region], levels = REG_LAB),
                           val = if (measure == "rate") rate else net,
                           lab = if (measure == "rate") sprintf("Region %+.1f%s", rate, ifelse(n_seeds < 3, "*", ""))
                                 else sprintf("Region total %+.0f%s", net, ifelse(n_seeds < 3, "*", "")))
  col_of <- c(Africa = 1, `South Asia` = 1, Americas = 2, MENA = 2, Europe = 3, `Asia-Pacific` = 3)
  pal <- setNames(c("#FF3B30", "#E8746C", "#B9908C", "#5E5E5E", "#8FA9C9", "#5E9BE6", "#40A6FF"), bins)
  xsc <- if (measure == "rate") scale_x_continuous(limits = c(-10, 10), breaks = seq(-10, 10, 5), labels = function(v) sprintf("%+d", v))
         else scale_x_continuous(trans = pseudo_log_trans(sigma = 1), limits = c(-40, 300), breaks = c(-30, -3, 0, 3, 30, 250),
                                 labels = function(v) ifelse(v > 0, paste0("+", v), as.character(v)))
  one_col <- function(k) {
    d <- cr |> filter(col_of[as.character(region_lab)] == k) |> droplevels()
    r <- rr |> filter(region_lab %in% levels(d$region_lab)) |> mutate(region_lab = factor(as.character(region_lab), levels = levels(d$region_lab)))
    bot <- d |> group_by(region_lab) |> summarise(ybot = levels(d$country)[min(as.integer(country))], .groups = "drop")
    r <- r |> left_join(bot, by = "region_lab")
    name_col <- ifelse(d$small[match(levels(d$country), d$country)], "#8C8C8C", "#111111")
    p <- ggplot(d, aes(x, country)) +
      geom_vline(xintercept = 0, colour = "#888888", linewidth = 0.3) +
      geom_segment(aes(x = 0, xend = x, yend = country), colour = "#BBBBBB", linewidth = 0.25) +
      geom_point(aes(fill = bin, size = events), shape = 21, colour = "#222222", stroke = 0.15) +
      scale_fill_manual(values = pal, drop = FALSE, name = if (measure == "rate") "Country class (per 100)" else "Country class (count)") +
      scale_size_area(max_size = 2.8, breaks = c(100, 1000, 5000), labels = comma, name = "Event cell-months", limits = c(0, max(cr$events))) +
      xsc + scale_y_discrete(expand = c(0, 0.6)) +
      facet_grid(region_lab ~ ., scales = "free_y", space = "free_y",
                 labeller = as_labeller(function(v) sub("South Asia", "South\nAsia", sub("Asia-Pacific", "Asia-\nPacific", v)))) +
      labs(x = NULL, y = NULL) +
      theme_minimal(base_size = 6.5) +
      theme(panel.grid.major.y = element_blank(), panel.grid.minor = element_blank(),
            axis.text.y = suppressWarnings(element_text(size = name_pt, colour = name_col)), axis.text.x = element_text(size = 5.8),
            strip.text.y = element_text(angle = -90, face = "bold", size = 6.5, lineheight = 0.8, margin = margin(1, 1, 1, 1)),
            strip.background = element_rect(fill = "#EEEEEE", colour = NA), panel.spacing.y = grid::unit(0.2, "lines"),
            plot.margin = margin(2, 3, 2, 2), legend.position = "none")
    if (measure == "rate") {
      clipped <- d |> filter(abs(rate) > 10, !small)
      p <- p + geom_vline(data = r, aes(xintercept = val), colour = "#1F1F1F", linewidth = 0.4, linetype = "22") +
        geom_text(data = clipped, aes(label = sprintf("%+.1f", rate), hjust = ifelse(rate > 0, 1.4, -0.4)), vjust = 0.35,
                  size = 5.5 * PT, fontface = "bold")
    }
    p + geom_label(data = r, aes(x = if (measure == "rate") val else 300, y = ybot, label = lab), inherit.aes = FALSE,
                   hjust = if (measure == "rate") -0.05 else 1, size = 5.5 * PT, fontface = "bold", fill = "white", label.size = 0,
                   label.padding = grid::unit(0.06, "lines"))
  }
  nmax <- max(table(col_of[as.character(cr$region_lab)]))
  ROW <- 0.098; LEGC <- 0.55; XT <- 0.16; AX <- 0.22
  H <- LEGC + XT + AX + nmax * ROW + 0.3
  png(f_png, width = W, height = H, units = "in", res = 600, bg = "white")
  grid::grid.newpage()
  inch <- function(v) grid::unit(v, "in")
  vp <- function(x, y, w, h) grid::viewport(x = inch(x), y = inch(y), width = inch(w), height = inch(h), just = c("left", "bottom"))
  print(one_col(1) + one_col(2) + one_col(3) + plot_layout(widths = c(1, 1, 1)), vp = vp(0, LEGC + XT, W, H - LEGC - XT))
  grid::grid.text(xlab, x = inch(W / 2), y = inch(LEGC + XT / 2), gp = grid::gpar(fontsize = 7.5))
  grab_leg <- function(g) g$grobs[[which(sapply(g$grobs, function(x) x$name) == "guide-box")]]
  legC <- grab_leg(ggplotGrob(one_col(1) + guides(fill = guide_legend(nrow = 1, title.position = "left", override.aes = list(size = 2.4), order = 1),
                                                  size = guide_legend(nrow = 1, title.position = "left", order = 2)) +
                                theme(legend.position = "bottom", legend.box = "vertical", legend.box.just = "left",
                                      legend.title = element_text(size = 7, face = "bold"), legend.text = element_text(size = 6.5, margin = margin(l = 1, r = 3)),
                                      legend.spacing.y = grid::unit(0.02, "cm"), legend.key.size = grid::unit(0.24, "cm"))))
  grid::pushViewport(vp(0, 0, W, LEGC)); grid::grid.draw(legC); grid::popViewport()
  dev.off()
  cat(sprintf("[fig] wrote %s (%.2f x %.2f in), %d countries, %d with < %d event cell-months (gray names)\n",
              f_png, W, H, nrow(cr), sum(cr$small), small_ev))
}
draw_countries_all(sub("\\.png$", "_print_countries_all_rate.png", f_out), "rate")
draw_countries_all(sub("\\.png$", "_print_countries_all_count.png", f_out), "count")
  })
}

# ===================== ukr_map: scripts/phase4_scaling/fig_ukraine_map.R (original body; single-figure mode only) =====================
if (ONE == "ukr_map") {
  cat("[figures.R] ukr_map <- fig_ukraine_map.R (original body)\n")
  local({
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
args <- PASS   # was: args <- commandArgs(trailingOnly = TRUE)
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
  })
}

# ===================== ukr_gain: scripts/replication_v5/fig_ukraine_alert_gain_arm.R (original body; single-figure mode only) =====================
if (ONE == "ukr_gain") {
  cat("[figures.R] ukr_gain <- fig_ukraine_alert_gain_arm.R (original body)\n")
  local({
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
args <- PASS   # was: args <- commandArgs(trailingOnly = TRUE)
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
  })
}
cat("[figures.R] done\n")
