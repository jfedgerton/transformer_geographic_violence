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
args <- commandArgs(trailingOnly = TRUE)
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
