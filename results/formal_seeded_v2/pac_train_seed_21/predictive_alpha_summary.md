# 3D Predictive Authority Alpha

## Overall
```text
          method     rmse  post_startup_rmse  tail_rmse  mean_error  max_error    energy   z_rmse  authority_active_fraction  authority_alpha_mean  authority_alpha_std  action_saturation_step_fraction  action_tv_mean  action_jerk_mean  alpha_raw_mean  alpha_uncertainty_std  rmse_gain_pct  post_startup_gain_pct
predictive_alpha 0.092391           0.085932   0.102107    0.081983   0.303333 31.989171 0.014503                        1.0              0.223475             0.247569                         0.130063        0.019614          0.003559        0.199547                    0.0            NaN                    NaN
```

## Train Metrics
```json
{
  "train_mse": 0.0066471886821091175,
  "train_mae": 0.04559091851115227,
  "teacher_positive_fraction": 0.43904760479927063,
  "pred_alpha_mean": 0.21721681952476501,
  "teacher_alpha_mean": 0.18765872716903687,
  "policy_architecture": "transformer",
  "history_len": 16,
  "temporal_dataset_mode": "precomputed",
  "positive_mae": 0.02914419211447239,
  "teacher_cache_path": "",
  "teacher_cache_loaded": false,
  "teacher_samples_total": 18900,
  "teacher_samples_used": 18900,
  "max_train_samples": 0
}
```

## Figures
- results/figures/formal_seeded_v2/pac_train_seed_21/predictive_alpha_summary.png