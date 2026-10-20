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
args <- commandArgs(trailingOnly = TRUE)
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
