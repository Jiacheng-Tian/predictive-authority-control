# Preregistered follow-up experiments (2026-09-30)

All four experiments below were approved before any code execution.  The
frozen v3 baseline (`runs/formal_true_mpc_v3/formal-v3-baseline`) and the
frozen v4 archive (`results/formal_v4/`) are never modified; every new run
writes to a new directory.  Evaluation protocol for all formal runs: the
same paired 120-episode grid (85 seen / 35 unseen) defined by
`build_episode_tasks(profile="formal")` with `config/pac_v4.yaml`
unchanged; model-seed-level paired effects use t(4) 95% CIs.

## P0 — TD3 target-backbone decay fix and round-1 retrain

**Hypothesis.** The v4 residual-RL round-1 results are corrupted by a
target-network aliasing bug: the target actor shared the online backbone
module object, so each delayed soft update scaled every frozen backbone
tensor by (1 - tau^2).  Empirically verified before the fix: seed-31000
round-1 checkpoint backbone weights are exactly 0.0825x the frozen v3
baseline, matching (1 - 0.005^2)^(199601/2) = 0.0825.

**Fix.** (1) target actor built from a deepcopy of the backbone;
(2) actor soft update restricted to `source.requires_grad` parameters;
(3) regression test `TargetBackboneDecouplingTests` (1000 gradient
updates leave every backbone tensor bitwise unchanged, both online and
target).  Commit `c685b84`.

**Preregistered procedure.** Retrain both affected arms for all five
model seeds (31000-31004), 200k env steps each, identical config:

- `runs/predictive_authority_v4/rl-round1-fixed` (residual_rl arm)
- `runs/predictive_authority_v4/rl-wm-round1-fixed` (wm_residual_rl arm,
  world model `runs/predictive_authority_v4/wm-formal-v2/world_model.pt`)

Then rerun the 24-combo formal grid (`--include-wm-rl`) against the
retrained checkpoints into a new run directory.  Acceptance check before
each formal run: the saved checkpoint backbone tensors are bitwise equal
to the frozen v3 baseline.

**Reporting.** Update the Residual PAC vs Supervised PAC comparison
(residual_rl_vs_v3_transformer) and wm arm comparisons from the new run;
the frozen archive numbers are superseded and flagged as bug-affected in
the manuscript.

## P1 — Fixed-alpha RMSE sweep (alpha in {0.25, 0.5, 0.75})

**Question.** How does tracking error vary across the constant blending
coefficient, in-distribution and out-of-distribution?

**Preregistered procedure.** Add `constant_alpha_0.25` and
`constant_alpha_0.75` probe methods to the formal grid (gated behind
`--alpha-sweep`, so the main 24-combo protocol is unchanged) and run the
same 120-episode paired grid with only the two probes + smc + mpc +
sspo + `constant_alpha` anchors.  Deliverable: RMSE-3D vs alpha curve,
seen and unseen blocks, 120-episode means with per-episode pairing
against the alpha = 0.5 anchor already measured in the P0 formal run.
No statistical claims are attached to the sweep; it is descriptive.

## P2 — Round-2 partial unfreeze (encoder_last + head)

**Hypothesis.** Allowing TD3 to adapt the last encoder block and the
backbone head on top of the fixed round-1 residual head improves
residual PAC further without disturbing the frozen-protocol guarantees
for the remaining backbone blocks.

**Preregistered procedure.** Consume `rl.unfreeze_blocks_round2:
[encoder_last, head]`: for round 2 exactly
`backbone.encoder.layers[-1]` and `backbone.head` become trainable
(unknown block names are rejected).  Round 2 requires resuming from the
*fixed* round-1 checkpoint of the same seed; 200k env steps; output
`runs/predictive_authority_v4/rl-round2`.  Evaluate with the formal grid
restricted to the residual arm (rl_round=2) against the same v3 backbone
anchors; preregistered comparison: residual_rl_round2_vs_v3_transformer
on rmse_3d per block with t(4) CIs.

## P3 — Direct-RL negative control arm

**Question.** How much of residual PAC's performance comes from the
structured architecture (frozen backbone + alpha blending + safety
stack) rather than from TD3 itself?

**Preregistered procedure.** New `DirectRLPolicy`: an MLP
(LayerNorm -> 256 -> 256 -> tanh) mapping the same 16x24 observation
window directly to six thruster commands in [-1, 1]^6.  No backbone, no
alpha, no gain/smoothing/rate-limit stack, no forced-primary override.
TD3 hyperparameters are shared with the residual arm from
`config/pac_v4.yaml`; the reward drops only the `w_delta_alpha` term
(no alpha exists); no warm-start (the stage-one dataset stores scalar
alpha actions with different semantics).  Replay buffer and twin critics
carry a 6-dim action.  Train five seeds for 200k env steps into
`runs/predictive_authority_v4/direct-rl`; evaluate on the same 120-episode
grid (`--include-direct-rl`), preregistered comparison:
direct_rl_vs_v3_transformer and direct_rl_vs_residual_rl on rmse_3d per
block with t(4) CIs.

**Expected direction (falsifiable).** Direct RL underperforms residual
PAC on the unseen block; if it matches or beats it, the architectural
prior is not doing the work claimed in the manuscript.
