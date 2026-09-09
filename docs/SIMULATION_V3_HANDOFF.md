# Simulation V3 Handoff

## Objective

Finish the minimum simulation-only v3 evidence package. Do not work on edge
deployment, hardware, OOD, ablations, or SSPO in this branch.

The only claim permitted after this work is simulation-only. The archived
`formal_seeded_v2` package is legacy one-step predictive v2 evidence and must
not be combined with any v3 number, table, figure, or confidence interval.

## Repository State

- Branch: `codex/true-mpc-experiment-completion`
- Committed HEAD: `f308533` (`fix: enforce v3 checkpoint artifact metadata`)
- The working tree is intentionally dirty. Do not reset, checkout, stash, or
  discard the following work:
  - Modified: `MANIFEST.csv`, `SHA256SUMS.txt`, `scripts/train_pac.py`,
    `src/pac/authority/dataset.py`, `src/pac/authority/evaluation.py`,
    `src/pac/authority/model.py`, `src/pac/evaluation/episodes.py`, and the
    related T5 tests.
  - New: `scripts/run_formal_v3.py`, `src/pac/evaluation/formal_v3.py`, and
    `tests/test_formal_v3.py`.
- Full regression executed on this exact dirty tree:

  ```powershell
  $env:PYTHONPATH=(Resolve-Path 'src').Path
  python -B -m unittest discover -s tests -q
  ```

  Result: `Ran 150 tests ... OK`. OSQP emits a non-fatal pending-deprecation
  warning.

## Completed Foundations

- T0: `config/pac_v3.yaml` defines separated train, model, evaluation, SSPO,
  and robustness seed partitions. The v3 model seeds are `31000..31004` and
  held-out evaluation seeds are `41000..41019`.
- T1: the common simulator actuator applies amplitude clipping then slew
  limiting. Metrics can use applied rather than requested commands.
- T2: `MPCController` is a 12-state, 20-step constrained LTV QP solved with
  OSQP. It has warm start, fallback telemetry, and the legacy one-step law is
  isolated in `legacy_predictive.py`.
- T3: immutable `EpisodeSpec` and causal current estimation provide paired
  initial states and current-estimate traces across methods.
- T4: the oracle generator creates one shared, content-addressed dataset,
  reusing each fresh MPC plan for an H=20, 11-alpha rollout oracle. Oracle
  generation has an offline 50 ms solver budget; online evaluation retains the
  7.5 ms QP budget.
- T5: v3 episode-level training, checkpoint provenance, and 5-seed training
  support exist. The uncommitted changes add RNG restoration, atomic artifacts,
  stronger checkpoint validation, and safer output paths.

## Current Runner

The uncommitted paired runner is intentionally small:

- Entry point: `scripts/run_formal_v3.py`.
- Engine: `src/pac/evaluation/formal_v3.py`.
- Methods: `real10kg_smc_steady`, `real10kg_mpc_ltv_v3`, and one PAC rollout
  per model seed.
- The formal grid is 5 model seeds x 20 environment seeds x 3 scenarios.
- The short grid is model seed `31000`, environment seed `41000`, scenario 1,
  and 20 steps.
- It writes a new child directory under a user-provided
  `runs/formal_true_mpc_v3` root and rejects output under `results`.
- Expected artifacts are `raw_metrics.csv`, `window_metrics.csv`,
  `overall_summary.csv`, `by_scenario.csv`, `paired_effects.csv`,
  `formal_summary.json`, `manifest.json`, and per-rollout timeseries CSVs.
- Episode metrics include 3-D RMSE, heading RMSE, applied control cost,
  saturation, actuator rate limiting, solver fallback fraction, solver deadline
  miss fraction, final/max error, and success at 1 m.
- The primary learned-method comparison must be paired by
  `(model_seed, scenario, environment_seed)`. The summary must report
  model-seed-level t confidence intervals, not a pooled rollout standard
  deviation as the primary uncertainty estimate.

## Next Actions

1. Inspect the existing dirty diff, finish or correct it, run the complete
   suite, refresh `MANIFEST.csv` and `SHA256SUMS.txt`, then commit it before
   starting any simulation run.
2. Review `formal_v3.py` for the following before treating it as ready:
   paired-key uniqueness; no evaluation seed derived from model seed; applied
   command metrics; separate baseline vs PAC rows; checkpoint and dataset hash
   validation; all output writes through a temporary directory plus atomic
   promotion; and no legacy-v2 read path except explicitly labeled references.
3. Generate a new oracle dataset in a fresh ignored run directory. First use
   `--profile short`, load it back, then generate the formal dataset only if the
   short artifact validates. Never reuse an archived v2 teacher table.
4. Train to a new ignored run directory. First use `--profile short`; formal
   must produce all five new checkpoints for seeds `31000..31004` from the same
   new dataset hash. Never use an archived v2 checkpoint.
5. Run the paired formal runner with `--profile short`. Verify all expected
   artifacts, paired keys, finite mandatory metrics, zero unintended fallback
   or deadline telemetry, and correct method/model dimensions.
6. Only after the short gate is clean, launch the formal grid. Keep its raw
   output below `runs/formal_true_mpc_v3/<run-id>`; do not promote it into
   `results` until all validation checks pass.
7. Add a single promotion/verification step that copies a validated immutable
   run to a new v3 evidence directory without touching legacy data. Its manifest
   must include commit, dirty state, config and dataset hashes, checkpoint
   hashes, dependency versions, seed map, and exact command.

## Commands

Use `PYTHONPATH=src` unless the package has been installed editable.

```powershell
$env:PYTHONPATH=(Resolve-Path 'src').Path

# Oracle short gate
python -B scripts/generate_oracle_dataset.py --profile short --out-dir runs/oracle-v3-short

# Training short gate
python -B scripts/train_pac.py --config pac_v3 --dataset-dir runs/oracle-v3-short --out-dir runs/train-v3-short --profile short

# Paired evaluation short gate
python -B scripts/run_formal_v3.py --config pac_v3 --dataset-dir runs/oracle-v3-short --checkpoints-dir runs/train-v3-short --output-root runs/formal_true_mpc_v3 --run-id short-gate --profile short
```

Before a formal run, commit the implementation so that the formal command does
not require `PAC_ALLOW_DIRTY_FORMAL=1`. A dirty formal run is diagnostic-only.

## Non-Negotiable Acceptance Criteria

- No overwrite of `results/formal_seeded_v2`, its checkpoints, figures, or
  timeseries pointers.
- Every new PAC result uses the shared fresh oracle dataset and a checkpoint
  with the matching dataset/config hash.
- SMC, true MPC, and PAC consume the same `EpisodeSpec` for each scenario and
  environment seed.
- Mandatory metrics are finite and derived from applied commands.
- The final summary labels the work as simulation-only and labels v2 as legacy
  one-step predictive evidence.
