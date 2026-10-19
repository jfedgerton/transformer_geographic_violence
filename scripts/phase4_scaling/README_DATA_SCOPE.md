# DATA SCOPE: READ BEFORE TRAINING ANY ENCODER (2026-10-19)

## What went wrong
- train_repr_obj.py used to default to REPR_VARIANT=holdout, which drops 26 named countries.
- That holdout was meant ONLY for the early loss/optimizer/statics test on a subset.
- It was carried into the S/M/L/XL size sweep (launch_scaleup.sbatch) and all 18 LORO encoders (launch_loro*.sbatch).
- So the transfer encoders were pretrained without Sudan, Ethiopia, Iraq, Pakistan, Myanmar, Ukraine, Mexico, Colombia and others.
- The LORO region filter also sat inside the holdout branch, so VARIANT=full would have leaked the target region.

## Rule
- LORO encoders exclude ONLY the target region: REPR_VARIANT=full, REPR_LORO=<region>.
- All-region encoders: REPR_VARIANT=full, no REPR_LORO.

## Enforced in code (train_repr_obj.py, patched 2026-10-19; original in archive_v8_2026-10-19/)
- REPR_VARIANT has no default; the run fails if it is unset.
- REPR_LORO requires REPR_VARIANT=full.
- The region filter applies under any variant; asserts no target-region rows remain.
- Writes data_model/repr_model_<TAG>.manifest.json (countries, rows by region, months, script md5).
- Asserts: under full, all non-target holdout-26 countries are present; train ends by 2020-12; val is 2021; no checkpoint overwrite.
- REPR_DRYRUN=1 writes the manifest to logs/ and exits. Run launch_dryrun_fullsrc.sbatch before any new training launcher.

## Superseded
- Encoders tagged L_contrastive_e_loro_* (no _fullsrc) and the S/M/L/XL sweep checkpoints were trained with the 26-country holdout.
