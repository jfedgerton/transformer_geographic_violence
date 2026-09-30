# Data provenance and licensing

This repository contains code only. Raw and intermediate data directories
(data_raw/, data_intermediate/, data_model/) are excluded via .gitignore
and are not redistributed here.

## UCDP (Uppsala Conflict Data Program)
License: CC BY 4.0. Redistribution is permitted with attribution.
Source: https://ucdp.uu.se/downloads/
Cite: Uppsala Conflict Data Program, Department of Peace and Conflict
Research, Uppsala University.

## ACLED (Armed Conflict Location & Event Data Project)
License: ACLED End User License Agreement. Redistribution of raw or
lightly-transformed ACLED data is NOT permitted. Access requires applying
directly at https://acleddata.com.
Source: https://acleddata.com/data-export-tool/

This repo's public/HF-release training arm (scripts/ucdp_release/) is
UCDP-only by design, enforced with a fail-loud assertion in
01_build_ucdp_data.py and 02_train_ucdp_repr.py that refuses to run on
any path containing "acled". The broader pipeline (scripts/00-05,
scripts/phase4_scaling/) uses ACLED for the main manuscript results and
is not licensed for redistribution of its data outputs.

## Other sources referenced in scripts/data_raw/
FEWS NET, GTD, GVA, PRIO-GRID, and supplementary datasets (ACD, CCC, EPR,
Fariss, GDELT, ICEWS, IDMC, mass mobilization, MPV, NAVCO, PITF, Polity,
PTS, SCAD, V-Dem, WGI) each carry their own source licenses. Consult the
original providers before redistributing anything derived from them.
