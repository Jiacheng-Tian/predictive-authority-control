# Erratum — v1.0.1

## Summary

The v1.0.0 release shipped residual-PAC checkpoints trained with a
TD3 target-network aliasing bug, and its headline tables and README
claims were produced by those corrupted checkpoints. The released
**code** already contained the fix (the bug was found and corrected
during pre-release verification), so retraining from the released code
never reproduced the released tables. v1.0.1 replaces the checkpoints,
all shipped result tables, and the README claims, and adds the
conservative deployment anchor that constitutes the final method.

## The bug

`TD3Trainer` constructed the target actor by passing it the **same
backbone module object** as the online policy. The soft update

```
target.mul_(1 - tau).add_(tau * source)
```

therefore aliased target and source onto identical tensors: after
`mul_`, the shared tensor already held `(1-tau) * w`, so the subsequent
`tau * source` read the scaled value, and each delayed update multiplied
every frozen backbone weight by `(1 - tau^2)`. With tau = 0.005,
policy_delay = 2 and 199,601 gradient updates, every backbone tensor in
the shipped checkpoints decayed to exactly **0.0825x** its frozen
supervised value (verified: median norm ratio 0.0825 == (1-tau^2)^{N}).

Architecturally, scaling all transformer weights mostly cancels inside
the LayerNorms; the net effect compressed the backbone's alpha output
toward ~0.5. The corrupted policy therefore behaved like a conservative
mid-blend controller, which happened to generalize well
out-of-distribution. The v1.0.0 headline numbers (OOD 0.116 m, −28%;
actuator-degradation 0.090 m, −69%) were an artifact of this accident,
not of residual learning.

## What changed in v1.0.1

| Item | v1.0.0 | v1.0.1 |
|---|---|---|
| `results/paper/checkpoints/rl_seed_*.pt` | decayed (0.0825x) | corrected (backbone bitwise identical to the shipped supervised checkpoints; asserted) |
| `results/paper/*.csv`, `formal_summary.json` | bug-affected run | corrected-protocol run (unanchored) |
| README headline | −28% / −69% (artifact) | corrected numbers + anchored default deployment |
| Deployment | raw alpha | conservative anchor `alpha' = 0.25 + 0.30*(alpha-0.25)` (default; `--no-deployment-anchor` reproduces the unanchored tables) |
| New | — | `results/paper/anchored/` (default-deployment tables + paired CIs) |

## Corrected results (120 paired episodes, 5 model seeds, t(4) CIs)

| RMSE-3D | Supervised PAC | Residual PAC (unanchored) | Residual PAC (anchored) |
|---|---|---|---|
| In-distribution | 0.0635 m | 0.0656 m | **0.0648 m** |
| Out-of-distribution | 0.1612 m | 0.1539 m | **0.1151 m** |
| Actuator degradation (OOD) | 0.3023 m | 0.3054 m | **0.0870 m** |

Key statistical statements (model-seed level paired effects):

- Anchored vs supervised, OOD: −0.0461 m, CI [−0.0736, −0.0186]
  (**−28.6%, significant**).
- Anchored vs the strongest fixed blending (constant alpha = 0.25,
  0.0713/0.1183 m): significantly better on **both** blocks
  (seen −0.0065 [−0.0072, −0.0058]; unseen −0.0032 [−0.0042, −0.0022]).
- Unanchored vs supervised: not significant on either block
  (unseen −0.0073 [−0.0302, +0.0156]).

## Interpretation

Under the corrected protocol, the residual stage alone does not
significantly improve out-of-distribution tracking. The conservative
deployment anchor — which keeps 30% of the learned policy's relative
judgement while anchoring the absolute authority level — recovers and
exceeds the v1.0.0 artifact's performance legitimately: it dominates
every fixed-blending baseline on both blocks. The anchor is the
deliberate, reproducible form of the accident that produced the
retracted v1.0.0 numbers.

## Verification

- Every shipped residual checkpoint's backbone tensors are bitwise
  identical to the shipped supervised checkpoint of the same seed
  (`scripts/verify_backbone_frozen.py`).
- The corrected tables come from the preregistered paired grid; the
  full experiment record (exploration stages, falsified alternatives,
  preregistration documents) is preserved on the
  `followup-experiments` and `backbone-v2` branches.
