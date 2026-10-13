R_DIR <- "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
df <- readRDS(file.path(R_DIR, "data_raw/dengue_brazil/denv_resid.RDS"))
cat("n_unique GEOM:", length(unique(df$GEOM)), "range:", range(df$GEOM), "\n")
cat("n_unique MUNICIP:", length(unique(df$MUNICIP)), "\n")
cat("n_unique STATE_NO:", length(unique(df$STATE_NO)), "\n")
# is GEOM constant within municipality? sample a few
library(dplyr)
chk <- df %>% group_by(MUNICIP) %>% summarise(n_geom = length(unique(GEOM))) %>% ungroup()
cat("municipalities with >1 GEOM value:", sum(chk$n_geom > 1), "of", nrow(chk), "\n")
# is GEOM just a row index within state, or a per-municipality running id?
sub <- df %>% filter(STATE == "AC") %>% distinct(MUNICIP, GEOM) %>% arrange(GEOM)
print(sub)
cat("[done]\n")
