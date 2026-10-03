# Evaluation Protocol

The evaluation is a **pre-registered, paired, simulation-only protocol**. All
comparisons, seed partitions, disturbance families, and acceptance criteria
were frozen in configuration files before the final training and evaluation
runs; the pipeline refuses to reuse or overwrite an existing run directory.

## Paired design

- **120 paired evaluation episodes** per method: identical episode
  realizations (same environment seeds, same disturbance draws) are replayed
  once per method, so every comparison is within-episode paired.
- **5 independently trained model seeds** per learned method.
- **6 compared methods**: `smc`, `mpc`, `constant_alpha` (fixed alpha = 0.5),
  `sspo`, `supervised_transformer` (supervised PAC), `residual_rl`
  (residual PAC, this work).

## Disturbance families

| Block | Family |
|---|---|
| seen | `actuator_delay_noise` |
| seen | `colored_noise` |
| seen | `mass_damping_mismatch` |
| seen | `ou_current` |
| seen | `random_freq_amp` |
| seen | `structured` |
| unseen | `actuator_delay_noise` |
| unseen | `colored_noise` |
| unseen | `estimation_delay` |
| unseen | `fast_ou` |
| unseen | `mass_damping_mismatch` |
| unseen | `ou_current` |
| unseen | `random_freq_amp` |

The `seen` block draws disturbance parameters from the training ranges; the
`unseen` block draws from held-out ranges (out-of-distribution).

## Metrics

`rmse_3d` (m), `heading_rmse_deg`, `applied_control_cost`,
`solver_deadline_miss_step_fraction`. Paired effects are reported with 95%
confidence intervals.

## Frozen artifact identifiers

For byte-level continuity with the frozen evidence chain, a small set of
internal protocol identifiers is intentionally kept unchanged inside saved
artifacts (checkpoint metadata such as `protocol_version`, the disturbance
family hash salt, and file-format version tags). These strings are
opaque format identifiers only and carry no versioning meaning for the
public API.
