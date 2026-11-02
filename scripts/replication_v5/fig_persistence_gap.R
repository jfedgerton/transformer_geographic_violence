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

args <- commandArgs(trailingOnly = TRUE)
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
