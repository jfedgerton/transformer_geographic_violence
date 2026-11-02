# Two-panel LORO figure for Section 4.1. Panel A gain over persistence (base+embedding minus persistence), panel B gain from the embedding (base+embedding minus base).
# Usage Rscript fig_loro_gains.R <bootstrap_ci_csv> <output_dir>
suppressPackageStartupMessages(library(tidyverse))
set.seed(123)
args <- commandArgs(trailingOnly = TRUE)
infile <- args[1]
outdir <- args[2]
stopifnot(file.exists(infile))
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
d <- read_csv(infile, show_col_types = FALSE) %>% filter(comparison %in% c("persistence", "nested"))
stopifnot(all(c("comparison", "region", "estimate", "lo95", "hi95") %in% names(d)))
stopifnot(nrow(d) == 14)
stopifnot(all(d$lo95 <= d$estimate), all(d$estimate <= d$hi95))
lab <- c(africa = "Africa", americas = "Americas", asia_pacific = "Asia-\nPacific", europe = "Europe", mena = "MENA", south_asia = "South\nAsia", ALL = "All regions")
cols <- c("All regions" = "#000000", "Africa" = "#E69F00", "Americas" = "#56B4E9", "Asia-\nPacific" = "#009E73", "Europe" = "#CC79A7", "MENA" = "#D55E00", "South\nAsia" = "#0072B2")
stopifnot(all(d$region %in% names(lab)))
order_regions <- d %>% filter(comparison == "nested", region %in% setdiff(names(lab), "ALL")) %>% arrange(region) %>% pull(region)
stopifnot(length(order_regions) == 6)
ypos <- c(setNames(6:1, order_regions), ALL = 7.5)
pa <- "A. Gain over persistence (PR-AUC difference)"
pb <- "B. Gain from the embedding (PR-AUC difference)"
d <- d %>% mutate(y = unname(ypos[region]), name = factor(unname(lab[region]), levels = names(cols)), panel = factor(ifelse(comparison == "persistence", pa, pb), levels = c(pa, pb)))
blank <- tibble(panel = factor(pa, levels = c(pa, pb)), x = c(0, 0.40), y = 1)
p <- ggplot(d, aes(y = y, colour = name)) +
  geom_vline(xintercept = 0, linewidth = 1.2, colour = "grey30", linetype = "dashed") +
  geom_hline(yintercept = 6.9, colour = "grey80", linewidth = 0.6) +
  geom_blank(data = blank, aes(x = x, y = y), inherit.aes = FALSE) +
  geom_linerange(aes(xmin = lo95, xmax = hi95), linewidth = 1.9) +
  geom_point(aes(x = estimate), size = 6.5) +
  scale_colour_manual(values = cols, guide = "none") +
  scale_y_continuous(breaks = unname(ypos), labels = unname(lab[names(ypos)]), expand = expansion(add = c(0.6, 0.6))) +
  facet_wrap(~panel, scales = "free_x") +
  labs(x = NULL, y = NULL) +
  theme_minimal(base_size = 16) +
  theme(strip.text = element_text(size = 22, hjust = 0, face = "bold"), axis.text = element_text(size = 19, colour = "black"), panel.grid.major.y = element_blank(), panel.grid.minor = element_blank(), panel.spacing.x = unit(1.4, "cm"), axis.line.x = element_line(colour = "black"))
ggsave(file.path(outdir, "fig_loro_gains.pdf"), p, width = 18, height = 6.2)
ggsave(file.path(outdir, "fig_loro_gains.png"), p, width = 18, height = 6.2, dpi = 300)
f3 <- function(x) sprintf("%.3f", x)
w <- d %>% mutate(txt = paste0(f3(estimate), " [", f3(lo95), ", ", f3(hi95), "]")) %>% select(comparison, region, txt) %>% pivot_wider(names_from = comparison, values_from = txt)
w <- w[match(names(ypos)[order(-ypos)], w$region), ]
stopifnot(nrow(w) == 7, w$region[1] == "ALL")
tex <- c("\\begin{tabular}{lcc}", "\\hline", "Region & Gain over persistence & Gain from embedding \\\\", "\\hline", paste0(gsub("\n", " ", lab[w$region]), " & ", w$persistence, " & ", w$nested, " \\\\"), "\\hline", "\\end{tabular}")
writeLines(tex, file.path(outdir, "tab_loro_gains_ci.tex"))
cat("[done]", file.path(outdir, "fig_loro_gains.pdf"), "\n")
