# Reproducibility

## Repository Contents

- `code/` contains the AUV environment, dynamics, thrusters, controllers, and
  Transformer authority model.
- `scripts/` contains the formal seeded protocol, SSPO calibration, manifest
  generation, and read-only repository verifier.
- `results/formal_seeded_v2/` contains five PAC training seeds, five
  checkpoints, formal raw metrics, fixed-controller metrics, and rollout data.
- `results/3d_authority_diagnosis/` contains the seed-20 SSPO mechanism study.

## Git LFS

The rollout timeseries CSV files are tracked by Git LFS. Install Git LFS before
working with a clone and fetch the data before verification:

```powershell
git lfs install
git lfs pull
```

## Validation Commands

Run these commands from the repository root after installing
`code/requirements.txt`:

```powershell
python scripts/run_formal_seeded_protocol.py --dry-run
python scripts/sspo_alpha_calibration.py --help
python scripts/update_release_manifest.py
python scripts/verify_repository.py
python -m unittest discover -s tests -v
```

The formal 5-seed, 180-epoch training run is not rerun by routine validation.
The protocol dry-run, checkpoint compatibility check, and short smoke workflow
provide executable coverage without replacing the archived formal evidence.
