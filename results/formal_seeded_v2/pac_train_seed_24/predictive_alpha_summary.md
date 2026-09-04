# 3D Predictive Authority Alpha

## Overall
```text
          method     rmse  post_startup_rmse  tail_rmse  mean_error  max_error    energy   z_rmse  authority_active_fraction  authority_alpha_mean  authority_alpha_std  action_saturation_step_fraction  action_tv_mean  action_jerk_mean  alpha_raw_mean  alpha_uncertainty_std  rmse_gain_pct  post_startup_gain_pct
predictive_alpha 0.090154           0.083116   0.092944    0.081267   0.268394 32.228917 0.014271                        1.0              0.231799             0.259286                         0.129317         0.01918          0.003464        0.203411                    0.0            NaN                    NaN
```

## Train Metrics
```json
{
  "train_mse": 0.006801558192819357,
  "train_mae": 0.04468679428100586,
  "teacher_positive_fraction": 0.43904760479927063,
  "pred_alpha_mean": 0.21852430701255798,
  "teacher_alpha_mean": 0.18765872716903687,
  "policy_architecture": "transformer",
  "history_len": 16,
  "temporal_dataset_mode": "precomputed",
  "positive_mae": 0.028415869921445847,
  "teacher_cache_path": "",
  "teacher_cache_loaded": false,
  "teacher_samples_total": 18900,
  "teacher_samples_used": 18900,
  "max_train_samples": 0
}
```

## Figures
- results/figures/formal_seeded_v2/pac_train_seed_24/predictive_alpha_summary.png