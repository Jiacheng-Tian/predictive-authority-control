# Follow-up experiment execution log (2026-09-30)

Machine: 20 logical cores, 32 GB RAM. Each training process pinned to
1 thread (`OMP_NUM_THREADS=1` + `torch.set_num_threads(1)`), ~1.6 GB RAM
each. Work dir: `D:\Nautilus\Predictive Authority Control`, branch `main`.

Preregistration: `docs/preregistered/2026-09-30_followups.md`.
Frozen, never touch: `runs/formal_true_mpc_v3/`, `results/formal_v4/`,
`runs/train-v3-formal-baseline/`, `runs/predictive_authority_v4/rl-round1/`,
`runs/predictive_authority_v4/rl-wm-round1/` (buggy but archived).

## P0 — TD3 backbone decay fix  [DONE]

- Bug verified: round-1 checkpoints' backbone = exactly 0.0825x frozen
  baseline = (1-tau^2)^(updates/policy_delay), tau=0.005,
  updates=199601, delay=2.
- Fix committed `c685b84`: deepcopy target backbone; soft update only
  `source.requires_grad`; regression test
  `TargetBackboneDecouplingTests` (1000 updates, bitwise stable).
- Smoke (8000 steps) verified: saved checkpoint backbone bitwise equal
  to frozen baseline after 6401 updates.

## P0 — Retraining  [IN PROGRESS]

- Batch A (residual_rl, 5 seeds) launched 11:47 local into
  `runs/predictive_authority_v4/rl-round1-fixed`; logs
  `runs/predictive_authority_v4/rl-fixed-seed3100*.txt`; background task
  id `exec_471ae571-f6ac-44d6-915b-4a993fc6356d`. ETA ~4-9 h
  (measured ~69%/core with 5 concurrent).
- Batch B (wm_residual_rl, 5 seeds) — run after A completes:
  ```
  for s in 31000 31001 31002 31003 31004; do OMP_NUM_THREADS=1 python \
    scripts/v4_train_rl.py \
    --dataset-dir runs/predictive_authority_v4/wm-dataset-formal-v2 \
    --out-dir runs/predictive_authority_v4/rl-wm-round1-fixed \
    --model-seed $s --wm-checkpoint \
    runs/predictive_authority_v4/wm-formal-v2/world_model.pt \
    > runs/predictive_authority_v4/rl-wm-fixed-seed$s.txt 2>&1 & done; wait
  ```
- P0 formal (after B): 24 combos x 120 episodes:
  ```
  OMP_NUM_THREADS=1 python scripts/v4_run_formal.py --profile formal \
    --dataset-dir runs/predictive_authority_v4/wm-dataset-formal-v2 \
    --rl-dir runs/predictive_authority_v4/rl-round1-fixed \
    --include-wm-rl --wm-rl-dir runs/predictive_authority_v4/rl-wm-round1-fixed \
    --wm-checkpoint runs/predictive_authority_v4/wm-formal-v2/world_model.pt \
    --sspo-schedule runs/predictive_authority_v4/sspo-v4/best_bias_schedule.json \
    --run-id formal-stage2-fixed-r1 --jobs 8
  ```
  Before it: verify every fixed checkpoint backbone bitwise == frozen
  baseline (script inline python, compare `backbone.*` vs
  `model_state_dict`).

## P1 — alpha sweep  [CODE DONE, committed]

- `--methods constant_alpha_0.25,constant_alpha_0.75` restricts grid to
  2 x 120 = 240 rollouts. Run (any time after nothing else heavy):
  ```
  OMP_NUM_THREADS=1 python scripts/v4_run_formal.py --profile formal \
    --methods constant_alpha_0.25,constant_alpha_0.75 \
    --dataset-dir runs/predictive_authority_v4/wm-dataset-formal-v2 \
    --rl-dir runs/predictive_authority_v4/rl-round1-fixed \
    --wm-checkpoint runs/predictive_authority_v4/wm-formal-v2/world_model.pt \
    --sspo-schedule runs/predictive_authority_v4/sspo-v4/best_bias_schedule.json \
    --run-id alpha-sweep-025-075 --jobs 8
  ```
- Deliverable: RMSE-3D vs alpha {0.25, 0.5, 0.75} for seen/unseen;
  alpha=0.5 anchor from the P0 formal run's constant_alpha rows (same
  episodes). Plot to `paper/generated_figures_v4/` (or results/figures).

## P2 — round 2  [CODE DONE, committed]

- `train_residual_rl` now: round 2 requires --resume-checkpoint; thaws
  exactly `encoder.layers[-1]` + `backbone.head` via
  `unfreeze_round2_blocks` (consumes rl.unfreeze_blocks_round2).
- Smoke first (small), then 5 seeds from the FIXED round-1 checkpoints:
  ```
  for s in ...; do OMP_NUM_THREADS=1 python scripts/v4_train_rl.py \
    --dataset-dir runs/predictive_authority_v4/wm-dataset-formal-v2 \
    --out-dir runs/predictive_authority_v4/rl-round2 --model-seed $s \
    --round 2 --resume-checkpoint \
    runs/predictive_authority_v4/rl-round1-fixed/residual_rl_seed_${s}_round1.pt \
    > runs/predictive_authority_v4/rl-round2-seed$s.txt 2>&1 & done; wait
  ```
- Eval: same formal command as P0 but `--rl-round 2 --rl-dir
  runs/predictive_authority_v4/rl-round2` (and decide whether to include
  wm arm; wm round-2 checkpoints do not exist, so plain run without
  --include-wm-rl unless trained).

## P3 — direct RL arm  [CODE DONE, committed; smoke IN PROGRESS]

- New `src/pac/v4/rl/direct.py`; train via `scripts/v4_train_direct_rl.py`
  (no dataset arg). 5 seeds:
  ```
  for s in ...; do OMP_NUM_THREADS=1 python scripts/v4_train_direct_rl.py \
    --out-dir runs/predictive_authority_v4/direct-rl --model-seed $s \
    > runs/predictive_authority_v4/direct-rl-seed$s.txt 2>&1 & done; wait
  ```
- Eval: formal command + `--include-direct-rl --direct-rl-dir
  runs/predictive_authority_v4/direct-rl` (can combine with P0 formal or
  run standalone with --methods).
- Reward for this arm drops only w_delta_alpha; no warmstart; replay
  starts empty; zero-init output layer (untrained = thrusters off).

## Notes / gotchas

- `tests.test_v4_runner` parity tests (v3 vs v4 solver deadline
  fraction; 1e-9 metric equality) are machine-timing sensitive and fail
  under CPU contention even on unmodified code (verified via git-stash
  control). Re-run on a quiet machine before trusting a failure.
- The `_worker_initialize` spawn path now carries 13 args (added
  direct_rl_dir, rl_round, alpha_sweep, include_direct_rl, methods).
- formal manifest records rl_round / alpha_sweep / methods_override /
  direct hashes.
- Post-training verification snippet for each checkpoint (P0 gate):
  compare state_dict `backbone.X` vs baseline `model_state_dict[X]`
  bitwise; `freeze_backbone` must be true in round-1 payload metadata.
