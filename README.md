# Predictive Authority Control

Predictive Authority Control (PAC) blends two complementary controllers for
autonomous underwater vehicle (AUV) trajectory tracking — a sliding-mode
primary controller (SMC) and a model-predictive authority controller (MPC) —
through a single bounded, learned authority coefficient. Instead of letting a
neural network output thrust commands directly, the learned component only
decides *how much to trust each controller* at every step:

- **Supervised PAC** — a compact Transformer (14,113 parameters) maps the most
  recent 16 steps of closed-loop history to the authority coefficient α ∈ [0, 1].
- **Residual PAC** — a zero-initialized residual head (2,177 parameters),
  trained with constrained TD3 reinforcement learning, may shift α by at most
  0.1 per step. At deployment start the system is numerically identical to the
  verified supervised baseline.

The complete learning system is 16,290 parameters (≈64 KB), small enough for
embedded deployment.

## Headline results

Paired, simulation-only evaluation over 120 episodes across eight disturbance
families (see [docs/evaluation_protocol.md](docs/evaluation_protocol.md)):

| Comparison | Supervised PAC | Residual PAC |
|---|---|---|
| Out-of-distribution RMSE | 0.161 m | **0.116 m (−28%)** |
| Actuator-degradation RMSE | 0.295 m | **0.090 m (−69%)** |
| In-distribution RMSE | **0.064 m** | 0.066 m (+4.4%) |

The in-distribution cost is disclosed deliberately: the residual stage trades
a small in-distribution regression for a large out-of-distribution gain.
Full tables and per-family breakdowns are shipped in
[results/paper](results/paper).

## Installation

Requires Python ≥ 3.11 with PyTorch ≥ 2.0.

```bash
python -m pip install -e .
# tested environment snapshot (torch 2.14.0, numpy 2.4.6, osqp 1.1.3, ...):
python -m pip install -r requirements-lock.txt
```

## Reproducing the pipeline

Each script documents its options with `--help`. All stages are deterministic
given the recorded seeds and refuse to overwrite existing outputs.

1. **Supervised stage**
   ```bash
   python scripts/generate_oracle_dataset.py   # rollout-oracle labels
   python scripts/train_pac.py                 # trains all model seeds
   python scripts/run_formal_supervised.py     # paired formal evaluation
   python scripts/sspo_alpha_calibration.py    # alpha-bias calibration arm
   ```
2. **Residual stage** (starts from the frozen supervised checkpoints)
   ```bash
   python scripts/train_residual_rl.py         # constrained TD3, one seed per call
   python scripts/run_formal_paired.py         # 120-episode paired grid
   python scripts/verify_backbone_frozen.py    # asserts backbone stayed frozen
   python scripts/summarize_formal_results.py  # tables from a run directory
   ```

Pre-trained checkpoints for both stages are included under
[results/paper](results/paper): five residual-PAC seeds (each checkpoint is
self-contained — backbone plus residual head) and the five frozen supervised
backbones they were initialized from, together with the frozen result tables.

## Repository layout

```
src/pac/
  simulation/    AUV dynamics, actuators, thrusters, observation model
  controllers/   SMC, MPC (QP-based), one-step predictive baseline, presets
  authority/     supervised PAC: features, oracle, Transformer, training
  residual/      residual PAC: disturbance families, TD3, reward, paired eval
  evaluation/    paired protocol, episode specs, metrics, summaries
scripts/         pipeline entry points (see above)
config/          frozen experiment configurations
tests/           unit and parity tests (pytest)
results/paper/   shipped checkpoints + frozen result tables
docs/            evaluation protocol
```

## Numerical compatibility

Results are deterministic for a fixed environment, but floating-point
summation order differs across torch versions and hardware. Expect small
numeric variation when re-running on a different stack; the tests encode
relaxed tolerances for cross-version comparison.

## Citation

See [CITATION.cff](CITATION.cff).

## License

MIT — see [LICENSE](LICENSE).
