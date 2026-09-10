# Predictive Authority Control

Predictive Authority Control (PAC) learns a bounded temporal authority
coefficient that blends explicit sliding-mode and MPC
controllers for 6-DOF AUV trajectory tracking in simulation.

## Evidence

### Legacy v2 evidence

The archived `results/formal_seeded_v2/` results are legacy one-step
predictive v2 evidence. They are not comparable with v3. The legacy formal
simulation evaluates five independently trained PAC models (training seeds
20-24) over ten evaluation episodes and three current scenarios. PAC achieves
aggregate 3-D position RMSE of `0.0902 +/- 0.0098 m`. This improves on the SMC
and MPC controllers by 37.2% and 38.1%, respectively. After SSPO calibration,
the seed-20 model reaches `0.07417 m`, a further 17.29% RMSE reduction from
its supervised PAC result.

## v3 simulation-only formal evidence

The v3 formal evidence is standalone, simulation-only evidence and is archived at
`results/formal_v3/formal-2026-09-10`.

The metric-specific mixed result versus `real10kg_mpc_ltv_v3` is that
PAC/predictive_alpha lowers `rmse_3d` by `0.00760 m` with model-seed t(4) CI
[-0.00794, -0.00726], but has higher `heading_rmse_deg` and
`solver_deadline_miss_step_fraction`.

All reported results are obtained in a 6-DOF AUV simulation. PAC blends an
explicit SMC primary controller with an MPC authority controller.

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
- `results/formal_v3/formal-2026-09-10/` contains the v3 simulation-only formal
  evidence.
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
