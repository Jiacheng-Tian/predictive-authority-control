# Preregistration E1: conservative-centering residual RL (2026-10-01)

Approved after the P0-P3 follow-ups.  Goal: recover the OOD performance
that the (now-fixed) TD3 decay bug had accidentally delivered, using a
legitimate, preregistered mechanism on the RL-residual path.

## Diagnosis motivating E1 (from completed, committed runs)

- The fixed run's entire unseen gap vs the archived buggy run lives in
  one family: `actuator_delay_noise` unseen (0.3054 fixed vs 0.0900
  buggy; all other unseen families equal or better).
- That family's reachable floor is alpha ~0.25-0.35: constant
  alpha=0.25 scores 0.0746 there (MPC 0.8492, SMC 0.1561).
- The fixed residual deploys alpha 0.643 there (backbone lowers it from
  ~0.85 but not far enough) while its authority is capped at
  delta_max=0.1 -- the head cannot reach the safe region.
- `random_freq_amp` unseen is alpha-insensitive (~0.395 for every
  method) and is excluded from optimization expectations.
- Seen-block RMSE is nearly flat in alpha (0.066-0.071 over 0.25-0.75),
  so a conservative prior costs little in-distribution.

## Preregistered grid (3 configs, everything else frozen)

| arm | rl.delta_max | reward.w_center | reward.version |
|-----|-------------:|----------------:|-----------------|
| E1a | 0.35 | 0.0  | pac_v4_reward_v1 |
| E1b | 0.35 | 0.5  | pac_v4_reward_v2_center |
| E1c | 0.25 | 0.5  | pac_v4_reward_v2_center |

New reward term (applied alpha after the filter stack):
`r -= w_center * (alpha - alpha_center)^2`, `alpha_center = 0.5`,
implemented as optional fields defaulting to zero so the frozen
protocol config is unchanged (verified: `pac_v4.yaml` loads with
w_center=0 and `compute_step_reward` returns bitwise-identical values).

Weight rationale: typical per-step reward is ~-1.5 (position+heading
dominated); w_center=0.5 costs 0.045/step at alpha=0.8 (~3%), enough to
anchor without pinning.

## Procedure

1. Train each arm x 5 model seeds (31000-31004), 200k env steps, from
   the frozen v3 backbones; outputs under
   `runs/predictive_authority_v4/e1{a,b,c}`.  Acceptance gate before
   any evaluation: backbone tensors bitwise equal to the baseline.
2. Selection (preregistered): per arm, run the formal grid with
   methods {constant_alpha, v3_transformer, residual_rl} (11 combos x
   120).  Winner = lowest seen-block residual RMSE subject to the hard
   gate seen <= 0.068.  **The 35 unseen episodes are not used for any
   selection decision**; the per-arm selection runs do report them but
   the choice is made on seen only.
3. Final: winner goes through the full 24-combo formal grid
   (`formal-e1-winner`) against the P0 anchors.
4. Success criteria (all three required):
   unseen RMSE <= 0.125 (alignment with the 0.116-0.121 band),
   seen RMSE <= 0.066,
   paired residual-vs-v3_transformer t(4) CI excludes 0 on unseen.
5. Git policy: interim arms and selection runs stay untracked under
   runs/; only the winning method's code/config/prereg/results are
   committed.

## Falsifiable expectation

E1b/E1c deploy lower alpha on actuator-delay-like conditions
(unseen alpha <= 0.45 on that family) and recover most of the gap;
E1a (range only, no prior) improves that family partially.  If none of
the three meets the success criteria, Stage 2 (uncertainty-conditioned
alpha_ref) is the preregistered follow-up.

## E1 outcome (2026-10-02) -- all arms failed the seen gate

| arm | seen | unseen | actuator_delay-unseen (alpha -> rmse) |
|-----|-----:|-------:|----------------------------------------|
| E1a | 0.0760 | 0.1446 | 0.581 -> 0.249 |
| E1b | 0.0773 | 0.1606 | 0.566 -> 0.209 |
| E1c | 0.0827 | 0.1787 | 0.306 -> 0.149 |

Diagnosis: the unconditional center drags benign families' alpha from
~0.85 to 0.57-0.63 (seen RMSE 0.09-0.11 on families that want high
alpha), while E1c proves the target behavior is reachable (alpha 0.31
on actuator-delay halves that family's error).  No arm committed.

## Stage 2 preregistration (amendment, before any Stage-2 training)

Distress-gated centering: the centering term activates only when the
current position error exceeds a gate, leaving benign operation
untouched:

    r -= w_center * (alpha - alpha_center)^2  if e_pos > center_gate_error_m
                                                       (else no term)

New optional config field `reward.center_gate_error_m` (default 0.0 =
always active, preserving the E1 semantics exactly).  Frozen protocol
unaffected (w_center=0).

| arm | delta_max | w_center | alpha_center | gate (m) |
|-----|----------:|---------:|-------------:|---------:|
| S2a | 0.25 | 1.0 | 0.30 | 0.15 |
| S2b | 0.35 | 1.0 | 0.30 | 0.15 |

Rationale: benign seen/unseen families operate at e~0.05-0.07 m (below
gate); actuator-delay-unseen at ~0.25-0.35 m and random_freq_amp (seen
and unseen) at ~0.2-0.4 m provide genuine distress experience inside
the training distribution, so the fallback is learned causally (large
error -> back off authority) rather than by family identity.

## Stage 2 outcome (2026-10-07) -- both arms failed the seen gate

| arm | seen | unseen | actuator-delay-unseen (alpha -> rmse) |
|-----|-----:|-------:|----------------------------------------|
| S2a | 0.0830 | 0.1735 | 0.333 -> 0.1388 |
| S2b | 0.1038 | 0.1905 | 0.340 -> 0.1336 |

Diagnosis: reward-side conservatism cannot be made sufficiently
conditional.  Both arms generalized the distress fallback into global
conservatism (benign seen families' alpha dragged to 0.53-0.68 vs the
0.85-0.90 they want), damaging alpha-loving families
(colored_noise unseen 0.148 vs baseline 0.062).  The conditional SHAPE
is learned correctly (S2a: actuator 0.33 / benign 0.63-0.68 /
structured 0.74) but the level is uniformly too low.

## Stage 3 preregistration (deployment-side alpha temperature, before any run)

Zero-training deployment mapping applied to the *raw* alpha of an
already-trained residual policy before the shared gain/filter stack:

    alpha_raw' = c + T * (alpha_raw - c),  c = 0.5

selected policy base: **S2a checkpoints** (most conditional shape).
This is the honest, preregistered form of the accidental mechanism the
TD3 bug implemented (Section P0): squash toward the mid-blend that the
P1 sweep shows is near-optimal OOD and nearly free ID.

| variant | T |
|---------|---|
| T040 | 0.40 |
| T025 | 0.25 |

## Stage 3 outcome (2026-10-07) -- both variants failed the seen gate

| variant | seen | unseen | verdict |
|---------|-----:|-------:|---------|
| T=0.40 | 0.0950 | 0.1604 | FAIL (gate 0.068) |
| T=0.25 | 0.0898 | 0.1401 | FAIL |

The mechanism hypothesis again half-validated: actuator-delay-unseen
reached 0.1294 (T=0.40, alpha 0.395) and 0.1358 (T=0.25), and T=0.25
even matched the archive on colored-noise-unseen (0.0631 at alpha
0.63).  But no temperature simultaneously preserves benign in-
distribution performance: every squash drags benign families (which
want alpha 0.85+) down with it, and the learned alpha's per-step
variance costs benign tracking (the archived buggy policy's alpha was
near-constant, i.e. temporally smooth -- a property none of E1/S2/S3
reproduces, and per-family results are erratic, e.g. colored-noise
SEEN 0.111 at alpha 0.42 under T=0.25).

## Final verdict (preregistered close-out)

Per the Stage-3 fallback clause: **the RL-residual path cannot align
the archived (buggy) numbers by legitimate, preregistered means**.
Ten configurations were trained/evaluated across three stages (E1a-c,
S2a-b bases; T=0.40/0.25 deployments); none passed the seen gate while
approaching the unseen target.  The paper reports the corrected
protocol results (P0 formal run `formal-stage2-fixed-r1`) as final.
Per the user's git policy, none of the E1/S2/S3 code, configs, or
results are committed; everything remains under untracked `runs/` and
in this working-tree document for the record.

## Stage 4 preregistration (2026-10-07, user-revised criterion)

The user has revised the acceptance criterion: **only unseen matters**
(in-distribution metrics carry no weight for this decision).  Revised
success: unseen <= 0.125 (archive band 0.116-0.121), paired-vs-v3
t(4) CI excludes 0 on unseen.  The seen gate is dropped.

Rationale from all prior stages: (i) unseen improves monotonically as
deployment conservatism deepens (T040 0.1604 -> T025 0.1401); (ii) the
residual gap to the archive is alpha temporal smoothness, and the P0
residual (dm 0.1) is the one variant whose alpha is backbone-smooth.

Preregistered zero-training grid (deployment only, base checkpoints
already trained and gated):

| arm | base ckpts | config | squash T | alpha_rate_limit |
|-----|-----------|--------|---------:|-----------------:|
| p0t30 | rl-round1-fixed | pac_v4.yaml | 0.30 | 0.0125 (default) |
| p0t15 | rl-round1-fixed | pac_v4.yaml | 0.15 | 0.0125 |
| s2rl4 | s2a | pac_v4_s2a + rl 0.004 | 0.25 | 0.004 |
| s2rl2 | s2a | pac_v4_s2a + rl 0.002 | 0.25 | 0.002 |

Selection: best unseen RMSE among the four arms (evaluated with
methods=residual_rl only, 600 rollouts each).  The winner then runs
the anchored formal (19 combos x 120) for the paired CI.  Disclosure:
the grid is small and preregistered; the final paper reports the
selected deployment configuration with the grid disclosed, and the
unseen CI is indicative given the selection.

Falsifiable expectation: if the smoothness hypothesis is right, p0t15
approaches the constant-0.5 floor minus dithering (unseen 0.118-0.130)
and s2rl2 improves on T025's 0.1401; if p0 arms beat s2 arms, the
backbone-smooth prior dominates the learned conditional shape.

## Stage 5 preregistration (2026-10-08, user-approved optimization fix)

Diagnosis from the training-curve investigation: the 1-step TD target
propagates an alpha decision's consequence over ~100 bootstrap steps
(credit assignment), and the raw critic input is unnormalized.  Fix
(zero protocol change to deployment):

- n-step returns (n=20) aggregated at insertion, per-row horizon h,
  bootstrap gamma**h (warm-start rows stay h=1);
- batch 256 -> 1024, updates_per_step 1 -> 2;
- LayerNorm over the flattened critic input.

Config: pac_v4_s5a.yaml (rl.n_step/critic_input_norm optional fields,
defaults preserve the frozen protocol bitwise; 28 unit tests incl.
chain math, done truncation, n=1 legacy semantics).

Smoke (seed 31000, 31.5k steps): reward -0.565 -> -0.491 -- first arm
with non-declining reward; critic loss bounded/oscillating.

Formal procedure: 5 seeds x 200k steps (runs/predictive_authority_v4/
s5a), gate = backbone bitwise frozen, then unseen-only selection eval
(user criterion) against p0t30 (0.1346), constant a0.25 (0.1183), v3
(0.1612).  Success: unseen <= 0.125.  Falsifiable expectation: if the
reward rise is more than dithering-tax removal, unseen improves over
0.1401; if not, S5 confirms the ceiling with cleaner optimization.
