# 3D Predictive Authority Alpha

## Overall
```text
          method     rmse  post_startup_rmse  tail_rmse  mean_error  max_error    energy  z_rmse  authority_active_fraction  authority_alpha_mean  authority_alpha_std  action_saturation_step_fraction  action_tv_mean  action_jerk_mean  alpha_raw_mean  alpha_uncertainty_std  rmse_gain_pct  post_startup_gain_pct
predictive_alpha 0.089015           0.082555   0.092398    0.079816   0.269834 32.312711 0.01482                        1.0              0.237955             0.259874                         0.131873        0.019116          0.003412        0.206407                    0.0            NaN                    NaN
```

## Train Metrics
```json
{
  "train_mse": 0.006908721756190062,
  "train_mae": 0.04529225453734398,
  "teacher_positive_fraction": 0.43904760479927063,
  "pred_alpha_mean": 0.21392004191875458,
  "teacher_alpha_mean": 0.18765872716903687,
  "policy_architecture": "transformer",
  "history_len": 16,
  "temporal_dataset_mode": "precomputed",
  "positive_mae": 0.030629457905888557,
  "teacher_cache_path": "",
  "teacher_cache_loaded": false,
  "teacher_samples_total": 18900,
  "teacher_samples_used": 18900,
  "max_train_samples": 0
}
```

## Figures
- results/figures/formal_seeded_v2/pac_train_seed_22/predictive_alpha_summary.png