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

args <- commandArgs(trailingOnly = TRUE)
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
