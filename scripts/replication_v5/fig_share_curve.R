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

args <- commandArgs(trailingOnly = TRUE)
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
