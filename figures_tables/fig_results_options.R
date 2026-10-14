# fig_results_options.R -- design options for Results 4.1-4.4 (ggplot2, print size 6.5 in wide, 600 dpi).
# Inputs (long format; mock_*.csv now, real Roar CSVs later with the same columns):
#   ladder.csv : region, seed, mode, model (persistence|base|emb_base), prauc
#   random.csv : region, seed, mode, gain_pre, gain_rand        (nested gains with pretrained vs random encoder)
#   share.csv  : region, seed, draw, share (%), arm (nested|finetune), gain
# Usage: Rscript fig_results_options.R <ladder.csv> <random.csv> <share.csv> <outdir> <mock 0|1>
set.seed(123)
suppressPackageStartupMessages({library(ggplot2); library(dplyr); library(readr); library(tidyr); library(patchwork); library(scales)})
a <- commandArgs(trailingOnly = TRUE); stopifnot(length(a) == 5)
ladder <- read_csv(a[1], show_col_types = FALSE); random <- read_csv(a[2], show_col_types = FALSE)
share <- read_csv(a[3], show_col_types = FALSE); outdir <- a[4]; MOCK <- a[5] == "1"
dir.create(outdir, showWarnings = FALSE)
stopifnot(all(c("region","seed","mode","model","prauc") %in% names(ladder)),
          all(c("region","seed","mode","gain_pre","gain_rand") %in% names(random)),
          all(c("region","seed","draw","share","arm","gain") %in% names(share)))
REG <- c(africa="Africa", americas="Americas", asia_pacific="Asia-Pacific", europe="Europe", mena="MENA", south_asia="South Asia")
PERS <- "#A6A6A6"; BASE <- "#4D4D4D"; EMB <- "#2E8FE0"; RAND <- "#9A9A9A"
MODEL_LAB <- c(persistence = "Persistence", base = "Base (gradient boosting)", emb_base = "Base + embedding")
W <- 6.5
th <- theme_minimal(base_size = 8) +
  theme(panel.grid.minor = element_blank(), panel.grid.major.y = element_blank(), strip.text = element_text(face = "bold", size = 8.5),
        axis.text = element_text(size = 7.5, colour = "#222222"), axis.title = element_text(size = 8),
        legend.position = "bottom", legend.title = element_blank(), legend.text = element_text(size = 7.5),
        plot.title = element_text(face = "bold", size = 9.5), plot.margin = margin(4, 6, 4, 4),
        panel.spacing.x = grid::unit(1.2, "lines"))
save <- function(p, name, h) {
  f <- file.path(outdir, paste0(name, ".png"))
  png(f, width = W, height = h, units = "in", res = 600, bg = "white"); print(p)
  if (MOCK) grid::grid.text("MOCK DATA – LAYOUT ONLY", rot = 25, gp = grid::gpar(fontsize = 34, col = "#E8E8E8A0", fontface = "bold"))
  dev.off(); cat("[fig]", f, "\n")
}
# order regions by mean temporal nested gain (largest at top)
ord <- ladder |> filter(mode == "Temporal") |> pivot_wider(names_from = model, values_from = prauc) |>
  group_by(region) |> summarise(g = mean(emb_base - base), .groups = "drop") |> arrange(g) |> pull(region)
lvl <- REG[ord]
lad <- ladder |> group_by(region, mode, model) |> summarise(prauc = mean(prauc), .groups = "drop") |>
  mutate(region_lab = factor(REG[region], levels = lvl), model = factor(model, levels = names(MODEL_LAB), labels = MODEL_LAB),
         mode = factor(mode, levels = c("Temporal", "Pooled")))
stopifnot(!anyNA(lad$region_lab), !anyNA(lad$model))
mcol <- setNames(c(PERS, BASE, EMB), MODEL_LAB)

# ---- 4.1 option A1: dot ladder (persistence -> base -> base + embedding) -------------------------------------------
wide <- lad |> pivot_wider(names_from = model, values_from = prauc)
pA1 <- ggplot(lad, aes(prauc, region_lab)) +
  geom_segment(data = wide, aes(x = .data[[MODEL_LAB[1]]], xend = .data[[MODEL_LAB[3]]], y = region_lab, yend = region_lab),
               colour = "#D0D0D0", linewidth = 1.6, inherit.aes = FALSE) +
  geom_point(aes(colour = model), size = 2.4) +
  scale_colour_manual(values = mcol) + facet_wrap(~ mode) +
  scale_x_continuous(limits = c(0, NA), expand = expansion(mult = c(0, .04))) +
  labs(x = "PR-AUC (held-out region)", y = NULL) + th
save(pA1, "opt_4.1_A1_dot_ladder", 2.9)

# ---- 4.1 option A2: grouped bars with the embedding step labelled ----------------------------------------------------
lab2 <- wide |> mutate(g = .data[[MODEL_LAB[3]]] - .data[[MODEL_LAB[2]]])
pA2 <- ggplot(lad, aes(region_lab, prauc, fill = model)) +
  geom_col(position = position_dodge(width = 0.78), width = 0.72) +
  geom_text(data = lab2, aes(region_lab, .data[[MODEL_LAB[3]]], label = sprintf("%+.3f", g)), inherit.aes = FALSE,
            vjust = -0.5, hjust = 0.1, size = 6.5 / .pt, colour = EMB, fontface = "bold", nudge_x = 0.26) +
  scale_fill_manual(values = mcol) + facet_wrap(~ mode, ncol = 1) +
  scale_x_discrete(labels = function(x) sub(" ", "\n", sub("-", "-\n", x))) +
  scale_y_continuous(limits = c(0, NA), expand = expansion(mult = c(0, .12))) +
  labs(x = NULL, y = "PR-AUC (held-out region)") + th + theme(panel.grid.major.x = element_blank(), panel.grid.major.y = element_line())
save(pA2, "opt_4.1_A2_grouped_bars", 4.4)

# ---- 4.2 option B1: nested gain with seed dots, temporal and pooled ---------------------------------------------------
ng <- ladder |> pivot_wider(names_from = model, values_from = prauc) |> mutate(gain = emb_base - base) |>
  mutate(region_lab = factor(REG[region], levels = lvl), mode = factor(mode, levels = c("Temporal", "Pooled")))
ngm <- ng |> group_by(region_lab, mode) |> summarise(gain = mean(gain), .groups = "drop")
pd <- position_dodge(width = 0.55)
pB1 <- ggplot(ng, aes(gain, region_lab, group = mode)) +
  geom_vline(xintercept = 0, colour = "#888888", linewidth = 0.35) +
  geom_point(aes(shape = mode), position = pd, size = 1.1, colour = "#7FB8EE", alpha = 0.9) +
  geom_point(data = ngm, aes(shape = mode), position = pd, size = 2.8, colour = EMB, fill = EMB, stroke = 0.9) +
  scale_shape_manual(values = c(Temporal = 16, Pooled = 1)) +
  scale_x_continuous(labels = function(v) sprintf("%+.2f", v)) +
  labs(x = "Nested gain in PR-AUC (base + embedding − base)", y = NULL,
       caption = "Large marker = mean of three encoder seeds; small markers = individual seeds") +
  th + theme(plot.caption = element_text(size = 6.5, colour = "#666666"))
save(pB1, "opt_4.2_B1_gain_seeds", 2.8)

# ---- 4.2 option B2: heatmap region x mode with values -----------------------------------------------------------------
pB2 <- ggplot(ngm, aes(mode, region_lab, fill = gain)) +
  geom_tile(colour = "white", linewidth = 1.2) +
  geom_text(aes(label = sprintf("%+.3f", gain)), size = 7.5 / .pt, fontface = "bold",
            colour = ifelse(ngm$gain > 0.018, "white", "#111111")) +
  scale_fill_gradient2(low = "#FF3B30", mid = "#F2F2F2", high = "#1F6FC0", midpoint = 0, limits = c(-0.03, 0.03), oob = squish,
                       labels = function(v) sprintf("%+.2f", v), name = "Nested gain") + guides(fill = guide_colourbar(title.position = "top", title.hjust = 0.5)) +
  labs(x = NULL, y = NULL) + th + theme(legend.title = element_text(size = 7.5), panel.grid = element_blank(),
                                        legend.key.width = grid::unit(1.2, "cm"), legend.key.height = grid::unit(0.25, "cm"))
save(pB2 + theme(plot.margin = margin(4, 140, 4, 140)), "opt_4.2_B2_heatmap", 2.8)

# ---- 4.3 option C1: pretrained vs random encoder (dumbbell); the gap is the transfer -----------------------------------
rm_ <- random |> group_by(region, mode) |> summarise(pre = mean(gain_pre), rnd = mean(gain_rand), .groups = "drop") |>
  mutate(region_lab = factor(REG[region], levels = lvl), mode = factor(mode, levels = c("Temporal", "Pooled")))
rl <- rm_ |> pivot_longer(c(pre, rnd), names_to = "enc", values_to = "gain") |>
  mutate(enc = factor(enc, levels = c("rnd", "pre"), labels = c("Random encoder", "Pretrained encoder")))
pC1 <- ggplot(rl, aes(gain, region_lab)) +
  geom_vline(xintercept = 0, colour = "#888888", linewidth = 0.35) +
  geom_segment(data = rm_, aes(x = rnd, xend = pre, y = region_lab, yend = region_lab), colour = "#BFDDF7", linewidth = 1.8, inherit.aes = FALSE) +
  geom_point(aes(shape = enc, colour = enc, fill = enc), size = 2.4, stroke = 0.8) +
  scale_shape_manual(values = c(21, 21)) + scale_colour_manual(values = c(RAND, EMB)) + scale_fill_manual(values = c("white", EMB)) +
  facet_wrap(~ mode) + scale_x_continuous(labels = function(v) sprintf("%+.2f", v)) +
  labs(x = "Nested gain in PR-AUC; the gap between markers is the gain from pretraining", y = NULL) + th
save(pC1, "opt_4.3_C1_dumbbell_random", 2.9)

# ---- 4.3 option C2: transfer gap (pretrained - random) with seed dots ------------------------------------------------
rs <- random |> mutate(tr = gain_pre - gain_rand, region_lab = factor(REG[region], levels = lvl),
                       mode = factor(mode, levels = c("Temporal", "Pooled")))
rsm <- rs |> group_by(region_lab, mode) |> summarise(tr = mean(tr), .groups = "drop")
pC2 <- ggplot(rsm, aes(tr, region_lab)) +
  geom_col(fill = EMB, width = 0.6) +
  geom_point(data = rs, colour = "#123E66", size = 0.9) +
  geom_vline(xintercept = 0, colour = "#888888", linewidth = 0.35) + facet_wrap(~ mode) +
  scale_x_continuous(breaks = seq(-0.02, 0.04, 0.01), labels = function(v) sprintf("%+.2f", v)) +
  labs(x = "Gain from pretraining in PR-AUC (pretrained − random encoder)", y = NULL) + th
save(pC2, "opt_4.3_C2_transfer_bars", 2.9)

# ---- 4.4 option D1: gain by training share, regions thin, mean bold with 95% band -----------------------------------
ARM <- c(nested = "Nested (base + embedding − base)", finetune = "Fine-tune (pretrained − scratch)")
sh <- share |> mutate(arm = factor(ARM[arm], levels = ARM), region_lab = factor(REG[region], levels = REG))
shr <- sh |> group_by(arm, region_lab, share) |> summarise(gain = mean(gain), .groups = "drop")
shm <- sh |> group_by(arm, share) |> summarise(m = mean(gain), se = sd(gain) / sqrt(n()), .groups = "drop")
pD1 <- ggplot() +
  geom_hline(yintercept = 0, colour = "#888888", linewidth = 0.35) +
  geom_ribbon(data = shm, aes(share, ymin = m - 1.96 * se, ymax = m + 1.96 * se), fill = "#BFDDF7") +
  geom_line(data = shr, aes(share, gain, group = region_lab), colour = "#9A9A9A", linewidth = 0.35) +
  geom_line(data = shm, aes(share, m), colour = EMB, linewidth = 1.1) +
  geom_point(data = shm, aes(share, m), colour = EMB, size = 1.6) +
  facet_wrap(~ arm, scales = "free_y") +
  scale_x_log10(breaks = c(0.25, 0.5, 1, 2.5, 5, 10), labels = function(v) paste0(v, "%")) +
  scale_y_continuous(labels = function(v) sprintf("%+.2f", v)) +
  labs(x = "Share of the held-out region's training data", y = "Gain in PR-AUC",
       caption = "Blue = mean across regions, seeds and draws (band = mean \u00b1 1.96 SE across held-out units; descriptive spread, not a confidence interval); gray = individual regions") +
  th + theme(panel.grid.major.y = element_line(), plot.caption = element_text(size = 6.5, colour = "#666666"))
save(pD1, "opt_4.4_D1_share_curves", 2.9)

# ---- 4.4 option D2: heatmap region x share, one panel per arm --------------------------------------------------------
pD2 <- ggplot(shr, aes(factor(share), factor(region_lab, levels = rev(REG)), fill = gain)) +
  geom_tile(colour = "white", linewidth = 0.8) +
  geom_text(aes(label = sprintf("%+.2f", gain)), size = 6 / .pt, colour = ifelse(abs(shr$gain) > 0.05, "white", "#111111")) +
  facet_wrap(~ arm) +
  scale_fill_gradient2(low = "#FF3B30", mid = "#F2F2F2", high = "#1F6FC0", midpoint = 0, limits = c(-0.08, 0.08), oob = squish,
                       labels = function(v) sprintf("%+.2f", v), name = "Gain in PR-AUC") + guides(fill = guide_colourbar(title.position = "top", title.hjust = 0.5)) +
  scale_x_discrete(labels = function(v) paste0(v, "%")) +
  labs(x = "Share of the held-out region's training data", y = NULL) +
  th + theme(panel.grid = element_blank(), legend.title = element_text(size = 7.5),
             legend.key.width = grid::unit(1.2, "cm"), legend.key.height = grid::unit(0.25, "cm"))
save(pD2, "opt_4.4_D2_share_heatmap", 3.1)

# ---- E: one compact figure for 4.1-4.4 (A1 temporal, B1, C1 temporal, D1 mean only) --------------------------------
tag <- function(p, t) p + labs(title = t)
e1 <- tag(ggplot(filter(lad, mode == "Temporal"), aes(prauc, region_lab)) +
  geom_segment(data = filter(wide, mode == "Temporal"), aes(x = .data[[MODEL_LAB[1]]], xend = .data[[MODEL_LAB[3]]], y = region_lab, yend = region_lab),
               colour = "#D0D0D0", linewidth = 1.4, inherit.aes = FALSE) +
  geom_point(aes(colour = model), size = 2) + scale_colour_manual(values = mcol) +
  scale_x_continuous(limits = c(0, NA), expand = expansion(mult = c(0, .04))) + labs(x = "PR-AUC", y = NULL) + th,
  "A. Persistence, base, base + embedding")
e2 <- tag(pB1 + labs(caption = NULL, x = "Nested gain (PR-AUC)") + theme(legend.position = "bottom"), "B. Nested gain")
e3 <- tag(ggplot(filter(rl, mode == "Temporal"), aes(gain, region_lab)) +
  geom_vline(xintercept = 0, colour = "#888888", linewidth = 0.35) +
  geom_segment(data = filter(rm_, mode == "Temporal"), aes(x = rnd, xend = pre, y = region_lab, yend = region_lab), colour = "#BFDDF7",
               linewidth = 1.6, inherit.aes = FALSE) +
  geom_point(aes(colour = enc, fill = enc), shape = 21, size = 2, stroke = 0.8) +
  scale_colour_manual(values = c(RAND, EMB)) + scale_fill_manual(values = c("white", EMB)) +
  scale_x_continuous(labels = function(v) sprintf("%+.2f", v)) + labs(x = "Nested gain (PR-AUC)", y = NULL) + th,
  "C. Pretrained vs random encoder")
e4 <- tag(ggplot(shm, aes(share, m, colour = arm, fill = arm)) +
  geom_hline(yintercept = 0, colour = "#888888", linewidth = 0.35) +
  geom_ribbon(aes(ymin = m - 1.96 * se, ymax = m + 1.96 * se), colour = NA, alpha = 0.25) +
  geom_line(linewidth = 0.9) + geom_point(size = 1.3) +
  scale_colour_manual(values = c(EMB, BASE)) + scale_fill_manual(values = c(EMB, BASE)) +
  scale_x_log10(breaks = c(0.25, 1, 5, 10), labels = function(v) paste0(v, "%")) +
  scale_y_continuous(labels = function(v) sprintf("%+.2f", v)) + labs(x = "Training share", y = "Gain (PR-AUC)") +
  th + theme(panel.grid.major.y = element_line()) + guides(colour = guide_legend(ncol = 1), fill = guide_legend(ncol = 1)),
  "D. Gain by training share")
pE <- (e1 | e2) / (e3 | e4)
save(pE, "opt_E_combined_4panel", 5.6)
