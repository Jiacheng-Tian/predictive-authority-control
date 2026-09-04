# 3D Predictive Authority Alpha

## Overall
```text
          method    rmse  post_startup_rmse  tail_rmse  mean_error  max_error    energy   z_rmse  authority_active_fraction  authority_alpha_mean  authority_alpha_std  action_saturation_step_fraction  action_tv_mean  action_jerk_mean  alpha_raw_mean  alpha_uncertainty_std  rmse_gain_pct  post_startup_gain_pct
predictive_alpha 0.08988           0.083085   0.093818    0.080154   0.276558 32.264816 0.014104                        1.0              0.232366             0.257371                         0.127524        0.018919          0.003453        0.203471                    0.0            NaN                    NaN
```

## Train Metrics
```json
{
  "train_mse": 0.006278767250478268,
  "train_mae": 0.04540703818202019,
  "teacher_positive_fraction": 0.43904760479927063,
  "pred_alpha_mean": 0.21447695791721344,
  "teacher_alpha_mean": 0.18765872716903687,
  "policy_architecture": "transformer",
  "history_len": 16,
  "temporal_dataset_mode": "precomputed",
  "positive_mae": 0.029819082468748093,
  "teacher_cache_path": "",
  "teacher_cache_loaded": false,
  "teacher_samples_total": 18900,
  "teacher_samples_used": 18900,
  "max_train_samples": 0
}
```

## Figures
- results/figures/formal_seeded_v2/pac_train_seed_23/predictive_alpha_summary.png