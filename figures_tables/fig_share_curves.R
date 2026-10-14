# fig_share_curves.R -- performance by share of target training data (0.25%-10%), from make_share_tables.py outputs.
# Inputs: nested_<dom>.csv, finetune_<dom>.csv (unit, share, mean, lo, hi, pos, runs, n_train, arm_a, arm_b)
#   nested: arm_a = base + embedding, arm_b = base; finetune: arm_a = fine-tuned pretrained, arm_b = scratch (PR-AUC)
# Usage: Rscript fig_share_curves.R <nested.csv> <finetune.csv> <domain label> <unit label> <out prefix>
set.seed(123)
suppressPackageStartupMessages({library(ggplot2); library(dplyr); library(readr); library(tidyr); library(patchwork)})
a <- commandArgs(trailingOnly = TRUE); stopifnot(length(a) == 5)
rd <- function(f, arm) read_csv(f, show_col_types = FALSE) |> mutate(arm = arm)
d <- bind_rows(rd(a[1], "nested"), rd(a[2], "finetune")) |> mutate(pct = 100 * share)
stopifnot(!anyNA(d$arm_a), !anyNA(d$arm_b), all(abs(d$mean - (d$arm_a - d$arm_b)) < 1e-9))
ARM <- c(nested = "Add embedding to base model", finetune = "Fine-tune pretrained vs train from scratch")
EMB <- "#2E8FE0"; BASE <- "#4D4D4D"
th <- theme_minimal(base_size = 8) +
  theme(panel.grid.minor = element_blank(), axis.text = element_text(size = 7.5, colour = "#222222"), axis.title = element_text(size = 8),
        strip.text = element_text(face = "bold", size = 8.5), legend.position = "bottom", legend.title = element_blank(),
        legend.text = element_text(size = 7.5), plot.title = element_text(face = "bold", size = 9.5), plot.margin = margin(4, 8, 4, 4),
        panel.spacing.x = grid::unit(1.2, "lines"))
xs <- scale_x_log10(breaks = c(0.25, 0.5, 0.75, 1, 2.5, 5, 10), labels = function(v) ifelse(v == 0.75, "", paste0(v, "%")))
# absolute performance, mean across held-out units
ab <- d |> group_by(arm, pct) |> summarise(a = mean(arm_a), b = mean(arm_b), .groups = "drop") |>
  pivot_longer(c(a, b), names_to = "which", values_to = "prauc") |>
  mutate(model = case_when(arm == "nested" & which == "a" ~ "Base + embedding", arm == "nested" ~ "Base",
                           which == "a" ~ "Pretrained, fine-tuned", TRUE ~ "Scratch"),
         model = factor(model, levels = c("Base", "Base + embedding", "Scratch", "Pretrained, fine-tuned")),
         arm = factor(ARM[arm], levels = ARM))
pA <- ggplot(ab, aes(pct, prauc, colour = model, shape = model)) +
  geom_line(linewidth = 0.8) + geom_point(size = 1.8) +
  scale_colour_manual(values = c(BASE, EMB, BASE, EMB)) + scale_shape_manual(values = c(16, 16, 17, 17)) +
  facet_wrap(~ arm) + xs + labs(x = NULL, y = "PR-AUC (mean across held-out units)", title = "A. Performance") + th +
  guides(colour = guide_legend(nrow = 1), shape = guide_legend(nrow = 1))
# gain, units thin + mean bold with interval across units
g <- d |> mutate(arm = factor(ARM[arm], levels = ARM))
gm <- g |> group_by(arm, pct) |> summarise(m = mean(mean), se = sd(mean) / sqrt(n()), npos = sum(mean > 0), n = n(), .groups = "drop")
pB <- ggplot() +
  geom_hline(yintercept = 0, colour = "#888888", linewidth = 0.35) +
  geom_line(data = g, aes(pct, mean, group = unit), colour = "#B5B5B5", linewidth = 0.3) +
  geom_ribbon(data = gm, aes(pct, ymin = m - 1.96 * se, ymax = m + 1.96 * se), fill = "#BFDDF7", alpha = 0.7) +
  geom_line(data = gm, aes(pct, m), colour = EMB, linewidth = 1.1) + geom_point(data = gm, aes(pct, m), colour = EMB, size = 1.6) +
  geom_text(data = gm, aes(pct, m, label = paste0(npos, "/", n)), vjust = -1.1, size = 6 / .pt, colour = "#333333") +
  facet_wrap(~ arm) + xs + scale_y_continuous(labels = function(v) sprintf("%+.2f", v)) +
  labs(x = paste0("Share of the held-out ", a[4], "'s training data"), y = "Gain in PR-AUC", title = "B. Gain",
       caption = paste0(a[3], ". Blue = mean across held-out ", a[4], "s (band = mean \u00b1 1.96 SE across held-out units; descriptive spread, not a confidence interval); gray = individual ", a[4],
                        "s.\nLabels = ", a[4], "s with a positive gain. Each point averages 3 seeds x 5 draws. Unlabelled tick = 0.75%.")) +
  th + theme(plot.caption = element_text(size = 6.5, colour = "#666666", hjust = 0), plot.caption.position = "plot")
p <- pA / pB
ggsave(paste0(a[5], "_share_curves.png"), p, width = 6.5, height = 5.4, dpi = 600, bg = "white")
cat("[fig] wrote", paste0(a[5], "_share_curves.png"), "\n")
print(as.data.frame(gm |> mutate(across(c(m, se), ~ round(.x, 4)))))
