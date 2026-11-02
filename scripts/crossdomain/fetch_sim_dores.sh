#!/bin/bash
# fetch_sim_dores.sh -- download Brazil SIM death records (DORES, ICD-10 era, 1996+) from the DATASUS public FTP.
# Raw .dbc files only; nothing is converted here. Never overwrites a non-empty file. Run from the login node (needs internet).
# 2026-10-20, cross-domain test C3.
set -u
D=/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast/data_raw/sim_brazil/dores
mkdir -p "$D"; cd "$D"
URL=ftp://ftp.datasus.gov.br/dissemin/publicos/SIM/CID10/DORES/
curl -s --max-time 60 -l "$URL" | tr -d '\r' | grep -E '^DO[A-Z]{2}[0-9]{4}\.dbc$' | sort > filelist.txt
echo "[list] $(wc -l < filelist.txt) files on server"
n=0; fail=0
while read f; do
  if [ -s "$f" ]; then continue; fi
  if curl -s --retry 3 --max-time 600 -o "$f.part" "$URL$f" && [ -s "$f.part" ]; then mv "$f.part" "$f"; n=$((n+1)); else rm -f "$f.part"; fail=$((fail+1)); echo "[fail] $f"; fi
done < filelist.txt
echo "[done] downloaded=$n failed=$fail total_on_disk=$(ls DO*.dbc 2>/dev/null | wc -l) size=$(du -sh . | cut -f1)"
