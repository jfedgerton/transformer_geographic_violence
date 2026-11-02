# Reproduces the country aggregation inside fig_cell_alerts.R and dumps it to CSV.
# Read-only: writes one new file, touches nothing else. 2026-10-30
set.seed(123)
suppressPackageStartupMessages({library(sf);library(dplyr);library(readr)})
args <- commandArgs(trailingOnly=TRUE); stopifnot(length(args)==3)
f_cells <- args[1]; f_ne <- args[2]; f_out <- args[3]
sf_use_s2(FALSE)
cells <- read_csv(f_cells, show_col_types=FALSE)
stopifnot(all(c('gid','region','d_tp','d_fp','events','lat','lon') %in% names(cells)), !anyDuplicated(cells$gid))
cells <- cells |> mutate(net = d_tp - d_fp)
tmp <- tempfile(); unzip(f_ne, exdir=tmp)
world <- st_read(list.files(tmp,pattern='\\.shp$',full.names=TRUE)[1], quiet=TRUE) |> select(NAME) |> filter(NAME != 'Antarctica')
st_crs(world) <- 4326
pts <- st_as_sf(cells, coords=c('lon','lat'), crs=4326, remove=FALSE)
hit <- st_intersects(pts, st_make_valid(world))
ci <- vapply(hit, function(i) if (length(i)) i[1] else NA_integer_, integer(1))
cat(sprintf('[chk] %d of %d centroids off-land\n', sum(is.na(ci)), length(ci)))
ci[is.na(ci)] <- st_nearest_feature(pts[is.na(ci),], world)
cells$country <- world$NAME[ci]
stopifnot(!anyNA(cells$country))
cr <- cells |> group_by(country) |> summarise(net=sum(net), events=sum(events), .groups='drop') |>
  filter(events >= 100) |> mutate(rate = 100*net/events) |> arrange(desc(events))
write_csv(cr, f_out)
t <- head(cr, 22)
cat(sprintf('[res] N=%d  top22 pos=%d neg=%d zero=%d\n', nrow(cr), sum(t$rate>0), sum(t$rate<0), sum(t$rate==0)))
cat('[res] top22 negatives:\n'); print(as.data.frame(t[t$rate<0, c('country','events','rate')]))
cat('[res] all negatives:', sum(cr$rate<0), 'of', nrow(cr), '\n')
