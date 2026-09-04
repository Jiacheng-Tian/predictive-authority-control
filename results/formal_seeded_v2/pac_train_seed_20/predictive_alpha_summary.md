# 3D Predictive Authority Alpha

## Overall
```text
          method     rmse  post_startup_rmse  tail_rmse  mean_error  max_error    energy   z_rmse  authority_active_fraction  authority_alpha_mean  authority_alpha_std  action_saturation_step_fraction  action_tv_mean  action_jerk_mean  alpha_raw_mean  alpha_uncertainty_std  rmse_gain_pct  post_startup_gain_pct
predictive_alpha 0.089676           0.082248   0.092393    0.079901   0.272898 32.422553 0.014571                        1.0              0.238617             0.257914                          0.12854        0.019396          0.003524        0.211243                    0.0            NaN                    NaN
```

## Train Metrics
```json
{
  "train_mse": 0.006927554961293936,
  "train_mae": 0.04688000679016113,
  "teacher_positive_fraction": 0.43904760479927063,
  "pred_alpha_mean": 0.22002309560775757,
  "teacher_alpha_mean": 0.18765872716903687,
  "policy_architecture": "transformer",
  "history_len": 16,
  "temporal_dataset_mode": "precomputed",
  "positive_mae": 0.028546245768666267,
  "teacher_cache_path": "",
  "teacher_cache_loaded": false,
  "teacher_samples_total": 18900,
  "teacher_samples_used": 18900,
  "max_train_samples": 0
}
```

## Figures
- results/figures/formal_seeded_v2/pac_train_seed_20/predictive_alpha_summary.png