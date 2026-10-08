# Follow-up experiment execution log (2026-09-30)

## PAUSED STATE (2026-10-07, user-requested pause)

Everything is stopped.  Stage-3 status: T=0.40 selection COMPLETE
(`formal/select-s3-t040`: seen 0.0950 FAIL gate 0.068, unseen 0.1604;
actuator-delay-unseen fixed to 0.1294 at alpha 0.395 -- mechanism
works; benign families degraded by the uniform level pull; new finding:
alpha TEMPORAL SMOOTHNESS, not level, separates the buggy run's benign
performance from ours).  T=0.25 selection PAUSED at 792/1320 rows.

Resume command when the user asks (Windows Git Bash):

```
cd "D:\Nautilus\Predictive Authority Control"
export PYTHONPATH="D:\Nautilus\Predictive Authority Control\src"
OMP_NUM_THREADS=1 python scripts/v4_run_formal.py \
  --config config/pac_v4_s2a.yaml --profile formal \
  --methods constant_alpha,v3_transformer,residual_rl \
  --rl-dir runs/predictive_authority_v4/s2a --alpha-squash 0.25 \
  --dataset-dir runs/predictive_authority_v4/wm-dataset-formal-v2 \
  --wm-checkpoint runs/predictive_authority_v4/wm-formal-v2/world_model.pt \
  --sspo-schedule runs/predictive_authority_v4/sspo-v4/best_bias_schedule.json \
  --run-id select-s3-t025 --jobs 12 --progress-every 40 \
  --resume-from "runs/predictive_authority_v4/formal/.select-s3-t025.tmp-hg2tgh7a/partial_rows.jsonl"
```

NOTE: a foreign editable install of `pac` (D:\Nautilus\pac-clone-a3,
created 2026-10-03) shadows this repo globally -- ALWAYS export
PYTHONPATH as above or imports resolve to the wrong codebase.

After t025 completes: preregistered verdict per
`2026-10-01_e1_centering.md` Stage 3; expected FAIL of the seen gate ->
honest-conclusion close-out, unless the user authorizes a Stage 4
(tighter deployment alpha rate-limit to test the smoothness
hypothesis; zero training).

## E1 conservative-centering residual RL (2026-10-01 afternoon)

Prereg: `2026-10-01_e1_centering.md`.  Diagnosis (committed data):
unseen gap = actuator_delay_noise family only (0.3054 vs 0.0900);
floor alpha~0.25 -> 0.0746; fixed residual alpha 0.643 there with
delta_max=0.1 authority.  Grid: E1a {dm 0.35, w_center 0},
E1b {dm 0.35, w_center 0.5}, E1c {dm 0.25, w_center 0.5}.

- Code: RewardConfig.w_center/alpha_center optional (default 0/0.5,
  frozen protocol unaffected, bitwise-verified); reward centering term
  on the applied alpha; configs config/pac_v4_e1{a,b,c}.yaml.
- E1b smoke passed (checkpoint records delta_max 0.35,
  reward pac_v4_reward_v2_center).
- Wave 1 (E1a x5 + E1b x5) -> runs/predictive_authority_v4/e1{a,b};
  wave 2 (E1c x5) after.  Selection per arm: formal run with methods
  constant_alpha,v3_transformer,residual_rl (11x120), choose on seen
  only (gate seen<=0.068).  Final: winner 24-combo formal.
- Wave-1 launch command pattern:
  `OMP_NUM_THREADS=1 python scripts/v4_train_rl.py --config
  config/pac_v4_e1X.yaml --dataset-dir
  runs/predictive_authority_v4/wm-dataset-formal-v2 --out-dir
  runs/predictive_authority_v4/e1X --model-seed $s`
- Acceptance before eval: v4_verify_backbone_frozen on every checkpoint.
- Success: unseen<=0.125, seen<=0.066, paired CI vs v3 excludes 0.
  Commit to git only after acceptance.

## Completed runs (2026-10-01)

- P0 retraining: `runs/predictive_authority_v4/rl-round1-fixed` and
  `rl-wm-round1-fixed` (5 seeds x 200k steps, 199601 updates each);
  acceptance gate passed on all 10 checkpoints (backbones bitwise equal
  to frozen baselines).
- P2 round 2: `runs/predictive_authority_v4/rl-round2` (5 seeds from the
  fixed round-1 checkpoints); gate passed (always-frozen tensors
  bitwise equal).
- P3 direct RL: `runs/predictive_authority_v4/direct-rl` (5 seeds,
  200k steps; reward improved from -103.8 untrained to -39..-54; seed
  31004 critic diverged and is still evaluated per protocol).
- P0 formal: `runs/predictive_authority_v4/formal/formal-stage2-fixed-r1`
  (24 combos x 120 episodes).
- P1 sweep: `runs/predictive_authority_v4/formal/alpha-sweep-025-075`
  (2 x 120) + curve `paper/generated_figures_v4/fig_v4_f6_alpha_sweep.*`.
- P3 formal: `runs/predictive_authority_v4/formal/direct-rl-control`
  (19 combos x 120: anchors + residual + direct).
- P2 formal: `runs/predictive_authority_v4/formal/round2-eval`
  (19 combos x 120, rl_round=2).

### Headline finding (P0)

With the correctly frozen backbone the residual-RL arms no longer show
a significant RMSE advantage over the supervised backbone in either
block (unseen mean difference -0.0073 m, t(4) CI [-0.0302, 0.0156]);
the archived buggy run's unseen advantage (0.116 vs 0.154 m) was an
artifact of the decaying backbone (effectively squashing predicted
alpha toward mid-blend during training).  Fixed-alpha 0.5 is the
strongest unseen method overall, consistent with the P1 sweep
(unseen RMSE: 0.118 at alpha=0.25, 0.121 at 0.5, 0.179 at 0.75; seen:
0.071 / 0.066 / 0.067).

---

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
