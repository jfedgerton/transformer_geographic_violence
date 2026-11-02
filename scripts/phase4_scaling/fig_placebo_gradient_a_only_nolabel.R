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
args <- commandArgs(trailingOnly = TRUE)
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
