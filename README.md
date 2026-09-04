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

This repository contains simulation evidence only. The predictive controller
is a one-step damped predictive law, not a finite-horizon MPC solver.

## Quick Start

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r code/requirements.txt
.venv/Scripts/python scripts/run_formal_seeded_protocol.py --dry-run
.venv/Scripts/python scripts/sspo_alpha_calibration.py --help
.venv/Scripts/python scripts/verify_repository.py
.venv/Scripts/python -m unittest discover -s tests -v
```

Full reproduction guidance and Git LFS setup are in `REPRODUCIBILITY.md`.
