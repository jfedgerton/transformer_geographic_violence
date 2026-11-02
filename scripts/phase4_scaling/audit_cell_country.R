# audit_cell_country.R -- 2026-10-27, read-only. Reads outputs/audit_cells_in.csv (audit_cells_export.py)
# and compares each cell's baseline_features country label with Natural Earth 50m borders by centroid.
# Writes outputs/audit_cell_country.csv and prints a mismatch summary. No input file is modified.
suppressPackageStartupMessages({library(sf); library(dplyr); library(readr)})
sf_use_s2(FALSE)
Z <- 'data_raw/shapefiles/ne_50m_admin_0/ne_50m_admin_0_countries.zip'
stopifnot(file.exists(Z), file.exists('outputs/audit_cells_in.csv'))
ne <- st_read(paste0('/vsizip/', Z), quiet = TRUE)
stopifnot('ADMIN' %in% names(ne))
ne <- ne['ADMIN']; names(ne)[1] <- 'ne_country'
cells <- read_csv('outputs/audit_cells_in.csv', show_col_types = FALSE)
stopifnot(!any(duplicated(cells$grid_id)))
pts <- st_as_sf(cells, coords = c('lon','lat'), crs = 4326, remove = FALSE)
hit <- st_join(pts, ne, join = st_intersects, left = TRUE)
out <- st_drop_geometry(hit) %>% mutate(match = !is.na(ne_country) & ne_country == country)
stopifnot(nrow(out) >= nrow(cells))
write_csv(out, 'outputs/audit_cell_country.csv')
cat('cells', nrow(cells), ' match', sum(out$match), ' mismatch', sum(!out$match & !is.na(out$ne_country)), ' no NE polygon (coast/ocean)', sum(is.na(out$ne_country)), '\n\n')
cat('top disagreements (label -> Natural Earth):\n')
print(as.data.frame(head(out %>% filter(!is.na(ne_country), !match) %>% count(country, ne_country, sort = TRUE), 40)))
cat('\nmismatch by labelled country:\n')
print(as.data.frame(head(out %>% filter(!is.na(ne_country)) %>% group_by(country) %>% summarise(cells = n(), mismatch = sum(!match)) %>% filter(mismatch > 0) %>% arrange(desc(mismatch)), 40)))
