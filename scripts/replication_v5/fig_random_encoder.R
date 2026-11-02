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

args <- commandArgs(trailingOnly = TRUE)
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
