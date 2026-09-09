# Predictive Authority Control

Predictive Authority Control (PAC) learns a bounded temporal authority
coefficient that blends explicit sliding-mode and one-step predictive
controllers for 6-DOF AUV trajectory tracking in simulation.

## Evidence

The formal simulation result evaluates five independently trained PAC models
(training seeds 20-24) over ten evaluation episodes and three current
scenarios. PAC achieves aggregate 3-D position RMSE of `0.0902 +/- 0.0098 m`.
Fixed controllers use the same ten unique episodes once per scenario. The
`0.07417 m` result is a seed-20 SSPO mechanism study and is not part of the
formal multi-seed result.

The `0.0098 m` value is the sample standard deviation pooled across all 150
PAC rollouts and three scenarios. The standard deviation across the five
training-seed mean RMSE values is `0.0013 m`. The SSPO mechanism study is
exploratory: seeds `20000-20002` were used during bias search and are also
included in its ten-seed evaluation, so `0.07417 m` is not a strictly held-out
estimate.

This repository contains simulation evidence only. The predictive controller
is a one-step damped predictive law, not a finite-horizon MPC solver.

## Quick Start

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -e .
.venv/Scripts/python scripts/run_formal_seeded_protocol.py --dry-run
.venv/Scripts/python scripts/sspo_alpha_calibration.py --dry-run
.venv/Scripts/python scripts/summarize_formal_results.py
.venv/Scripts/python scripts/verify_repository.py
.venv/Scripts/python -m unittest discover -s tests -v
```

`requirements-lock.txt` records the package versions used for the current
repository verification. It is not presented as the original training
environment, which was not preserved.

## Repository Contents

- `src/pac/` contains the installable simulation, controller, Transformer,
  training, and evaluation modules.
- `config/pac.yaml` is the single runtime source for formal and SSPO settings.
- `scripts/` contains the formal seeded protocol, SSPO calibration, manifest
  generation, and read-only repository verifier.
- `results/formal_seeded_v2/` contains five PAC training seeds, five
  checkpoints, formal raw metrics, fixed-controller metrics, and rollout data.
- `results/3d_authority_diagnosis/` contains the seed-20 SSPO mechanism study.

## Git LFS

The rollout timeseries CSV files are tracked by Git LFS. Before validating a
clone, install Git LFS and fetch its data:

```powershell
git lfs install
git lfs pull
```

The formal 5-seed, 180-epoch training protocol is not rerun by routine
validation. The protocol dry-run, checkpoint compatibility check, and short
smoke workflow provide executable coverage without replacing the archived
formal evidence.

New experiments are written under the ignored `runs/` directory. The runners
refuse to use a non-empty output directory, so the archived evidence under
`results/` is never overwritten by default.
