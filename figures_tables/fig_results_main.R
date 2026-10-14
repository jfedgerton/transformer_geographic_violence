# fig_results_main.R -- main-text versions (temporal / Mode B only) of the 4.1-4.3 figures, real Roar results (Oct 14).
# Inputs: real_ladder_incidence.csv (region, seed, mode, model, prauc), real_random_incidence.csv (region, seed, mode, gain_pre, gain_rand)
# Pooled versions stay in fig_results_options.R for the appendix.
set.seed(123)
suppressPackageStartupMessages({library(ggplot2); library(dplyr); library(readr); library(tidyr); library(patchwork)})
a <- commandArgs(trailingOnly = TRUE); stopifnot(length(a) == 3)
lad <- read_csv(a[1], show_col_types = FALSE) |> filter(mode == "Temporal")
rnd <- read_csv(a[2], show_col_types = FALSE) |> filter(mode == "Temporal")
out <- a[3]
stopifnot(nrow(lad) == 54, nrow(rnd) == 18)
REG <- c(africa = "Africa", americas = "Americas", asia_pacific = "Asia-Pacific", europe = "Europe", mena = "MENA", south_asia = "South Asia")
PERS <- "#A6A6A6"; BASE <- "#4D4D4D"; EMB <- "#2E8FE0"; RAND <- "#8C8C8C"
MODEL_LAB <- c(persistence = "Persistence", base = "Base (gradient boosting)", emb_base = "Base + embedding")
W <- 6.5
th <- theme_minimal(base_size = 8) +
  theme(panel.grid.minor = element_blank(), panel.grid.major.y = element_blank(),
        axis.text = element_text(size = 7.5, colour = "#222222"), axis.title = element_text(size = 8),
        legend.position = "bottom", legend.title = element_blank(), legend.text = element_text(size = 7.5),
        plot.title = element_text(face = "bold", size = 9.5), plot.margin = margin(4, 8, 4, 4))
save <- function(p, name, h) { f <- file.path(out, paste0(name, ".png")); ggsave(f, p, width = W, height = h, dpi = 600, bg = "white"); cat("[fig]", f, "\n") }
wide <- lad |> pivot_wider(names_from = model, values_from = prauc) |> mutate(gain = emb_base - base)
ord <- wide |> group_by(region) |> summarise(g = mean(gain)) |> arrange(g) |> pull(region)
lvl <- REG[ord]
wm <- wide |> group_by(region) |> summarise(across(c(persistence, base, emb_base, gain), mean), .groups = "drop") |>
  mutate(region_lab = factor(REG[region], levels = lvl))
lm_ <- wm |> pivot_longer(c(persistence, base, emb_base), names_to = "model", values_to = "prauc") |>
  mutate(model = factor(model, levels = names(MODEL_LAB), labels = MODEL_LAB))
mcol <- setNames(c(PERS, BASE, EMB), MODEL_LAB)
fmt <- function(v) sprintf("%+.2f", v)

# M1 (4.1): dot ladder, temporal only, gain printed at right
pM1 <- ggplot(lm_, aes(prauc, region_lab)) +
  geom_segment(data = wm, aes(x = persistence, xend = emb_base, y = region_lab, yend = region_lab), colour = "#D9D9D9", linewidth = 1.8, inherit.aes = FALSE) +
  geom_point(aes(colour = model), size = 2.6) +
  geom_text(data = wm, aes(x = 0.97, y = region_lab, label = sprintf("%+.3f", gain)), inherit.aes = FALSE, hjust = 1, size = 7 / .pt,
            colour = EMB, fontface = "bold") +
  annotate("text", x = 0.97, y = length(lvl) + 0.6, label = "Embedding step", hjust = 1, size = 6.5 / .pt, colour = "#555555") +
  scale_colour_manual(values = mcol) +
  scale_x_continuous(limits = c(0, 0.97), breaks = seq(0, 0.9, 0.1), expand = expansion(mult = c(0, .01))) +
  scale_y_discrete(expand = expansion(add = c(0.6, 1))) +
  labs(x = "PR-AUC in the held-out region (temporal test, mean of three encoder seeds)", y = NULL) + th
save(pM1, "main_4.1_ladder_temporal", 2.6)

# M2 (4.2): nested gain, seed dots + mean
gm <- wide |> mutate(region_lab = factor(REG[region], levels = lvl))
pM2 <- ggplot(gm, aes(gain, region_lab)) +
  geom_vline(xintercept = 0, colour = "#888888", linewidth = 0.35) +
  geom_point(colour = "#8CC2F0", size = 1.6) +
  geom_point(data = wm, colour = EMB, size = 3.2, shape = 18) +
  geom_text(data = wm, aes(label = sprintf("%+.3f", gain)), nudge_y = 0.32, size = 6.5 / .pt, colour = EMB, fontface = "bold") +
  scale_x_continuous(labels = fmt, breaks = seq(-0.01, 0.05, 0.01)) +
  labs(x = "Nested gain in PR-AUC (base + embedding − base), temporal test", y = NULL,
       caption = "Diamond = mean of three encoder seeds; dots = individual seeds") +
  th + theme(plot.caption = element_text(size = 6.5, colour = "#666666"))
save(pM2, "main_4.2_gain_temporal", 2.5)

# M3 (4.3): pretrained vs random encoder, seed dots for both; gap = gain from pretraining
rl <- rnd |> pivot_longer(c(gain_pre, gain_rand), names_to = "enc", values_to = "gain") |>
  mutate(enc = factor(enc, levels = c("gain_rand", "gain_pre"), labels = c("Random encoder", "Pretrained encoder")),
         region_lab = factor(REG[region], levels = lvl))
rm_ <- rl |> group_by(region_lab, enc) |> summarise(gain = mean(gain), .groups = "drop")
rw <- rm_ |> pivot_wider(names_from = enc, values_from = gain)
pd <- position_nudge(y = 0)
pM3 <- ggplot(rl, aes(gain, region_lab)) +
  geom_vline(xintercept = 0, colour = "#888888", linewidth = 0.35) +
  geom_segment(data = rw, aes(x = `Random encoder`, xend = `Pretrained encoder`, y = region_lab, yend = region_lab), colour = "#C6E0F7",
               linewidth = 2, inherit.aes = FALSE) +
  geom_point(aes(colour = enc), size = 1.1, alpha = 0.8, position = position_nudge(y = -0.22)) +
  geom_point(data = rm_, aes(colour = enc, fill = enc), shape = 21, size = 2.8, stroke = 0.9) +
  scale_colour_manual(values = c(RAND, EMB)) + scale_fill_manual(values = c("white", EMB)) +
  scale_x_continuous(labels = fmt, breaks = seq(-0.03, 0.05, 0.01)) +
  labs(x = "Nested gain in PR-AUC, temporal test; the gap between the large markers is the gain from pretraining", y = NULL,
       caption = "Large markers = mean of three seeds; small markers below = individual seeds") +
  th + theme(plot.caption = element_text(size = 6.5, colour = "#666666"))
save(pM3, "main_4.3_random_temporal", 2.7)

# M4: combined 4.2 + 4.3 in one panel (pretrained gain with seed dots, random encoder as hollow marker)
pM4 <- ggplot(rl, aes(gain, region_lab, colour = enc)) +
  geom_vline(xintercept = 0, colour = "#888888", linewidth = 0.35) +
  geom_point(size = 1.1, alpha = 0.75, position = position_dodge2(width = 0.5, preserve = "single")) +
  geom_point(data = rm_, aes(fill = enc), shape = 21, size = 3, stroke = 0.9, position = position_dodge(width = 0.5)) +
  scale_colour_manual(values = c(RAND, EMB)) + scale_fill_manual(values = c("white", EMB)) +
  scale_x_continuous(labels = fmt, breaks = seq(-0.03, 0.05, 0.01)) +
  labs(x = "Nested gain in PR-AUC (base + embedding − base), temporal test", y = NULL) + th
save(pM4, "main_4.2-4.3_combined_temporal", 2.7)

# M5: three-panel main-text figure (A ladder, B gain, C random) stacked
pM5 <- (pM1 + labs(title = "A. Persistence, base and base + embedding", x = "PR-AUC")) /
       (pM3 + labs(title = "B. Gain from the embedding: pretrained vs random encoder", x = "Nested gain in PR-AUC", caption = NULL)) +
  plot_layout(heights = c(1, 1))
save(pM5, "main_E_two_panel_temporal", 4.8)
