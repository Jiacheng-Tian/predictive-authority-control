# Predictive Authority Control

Predictive Authority Control (PAC) learns a bounded temporal authority
coefficient that blends explicit sliding-mode and one-step predictive
controllers for 6-DOF AUV trajectory tracking in simulation.

## Evidence

The formal simulation result evaluates five independently trained PAC models
(training seeds 20-24) over ten evaluation episodes and three current
scenarios. PAC achieves aggregate 3-D position RMSE of `0.0902 +/- 0.0098 m`.
This improves on the SMC and one-step predictive controllers by 37.2% and
38.1%, respectively. After SSPO calibration, the seed-20 model reaches
`0.07417 m`, a further 17.29% RMSE reduction from its supervised PAC result.

All reported results are obtained in a 6-DOF AUV simulation. PAC blends an
explicit SMC primary controller with a lightweight one-step predictive
authority controller.

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

`requirements-lock.txt` records the package versions used for repository
verification and repeatable installation.

## Repository Contents

- `src/pac/` contains the installable simulation, controller, Transformer,
  training, and evaluation modules.
- `config/pac.yaml` is the single runtime source for formal and SSPO settings.
- `scripts/` contains the formal seeded protocol, SSPO calibration, manifest
  generation, and read-only repository verifier.
- `results/formal_seeded_v2/` contains five PAC training seeds, five
  checkpoints, formal raw metrics, fixed-controller metrics, and rollout data.
- `results/3d_authority_diagnosis/` contains the seed-20 SSPO results.

## Git LFS

The rollout timeseries CSV files are tracked by Git LFS. Before validating a
clone, install Git LFS and fetch its data:

```powershell
git lfs install
git lfs pull
```

Routine validation covers the formal protocol, all five checkpoints, the
headline metrics, and a short end-to-end training and evaluation workflow.

New experiments are written under the ignored `runs/` directory. The runners
refuse to use a non-empty output directory, so the archived evidence under
`results/` is never overwritten by default.
