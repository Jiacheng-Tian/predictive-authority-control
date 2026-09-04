# 3D Authority Diagnosis

This diagnostic package evaluates fixed 3D baselines only. It is intended to locate authority intervention windows before training a neural policy.

## Metadata
```json
{
  "steps": 2100,
  "scenarios": [
    1,
    2,
    3
  ],
  "seeds": [
    20000,
    20001,
    20002,
    20003,
    20004,
    20005,
    20006,
    20007,
    20008,
    20009
  ],
  "base_controllers": [
    "real10kg_smc_steady",
    "real10kg_mpc_event"
  ],
  "current_amplitude_scales": [
    2.5
  ],
  "current_frequency_scales": [
    1.0
  ],
  "initial_position_std": 0.03,
  "initial_velocity_std": 0.01,
  "vertical_current": 0.75,
  "vehicle_profile": "real_10kg_v1",
  "action_mode": "thruster",
  "thruster_layout": "real_10kg_x",
  "start_times": [
    0.0
  ],
  "mass_scale_xy": 1.0,
  "damping_scale_xy": 1.0,
  "reference_controller": "real10kg_smc_steady",
  "trajectory3d": true
}
```

## Overall Metrics
```text
         controller  current_amplitude_scale     rmse  post_startup_rmse   z_rmse  action_jerk_mean  action_saturation_step_fraction
real10kg_smc_steady                 2.500000 0.143748           0.122230 0.021379          0.001178                         0.112159
 real10kg_mpc_event                 2.500000 0.145713           0.150004 0.008412          0.001660                         0.256778
```

## Scenario Metrics
```text
         controller    scenario  current_amplitude_scale     rmse  post_startup_rmse   z_rmse  action_jerk_mean
real10kg_smc_steady    constant                 2.500000 0.147065           0.127233 0.021345          0.000252
 real10kg_mpc_event    constant                 2.500000 0.153032           0.161792 0.008337          0.001263
real10kg_smc_steady  sinusoidal                 2.500000 0.149973           0.130054 0.021519          0.000267
 real10kg_mpc_event  sinusoidal                 2.500000 0.244819           0.262405 0.008435          0.001427
 real10kg_mpc_event step_change                 2.500000 0.039287           0.025816 0.008465          0.002290
real10kg_smc_steady step_change                 2.500000 0.134205           0.109401 0.021273          0.003017
```

## Window Axis Metrics
```text
         controller               window  rmse_3d  xy_rmse   z_rmse  z_energy_share  window_action_jerk_mean
 real10kg_mpc_event         startup_0_3s 0.081294 0.079773 0.015458        0.038147                 0.007716
 real10kg_mpc_event       pre_step_3_10s 0.081794 0.081587 0.003355        0.019699                 0.000189
 real10kg_mpc_event step_recovery_10_13s 0.312909 0.312663 0.007638        0.008105                 0.002384
 real10kg_mpc_event   post_step_13_20p9s 0.071528 0.070374 0.007959        0.134916                 0.000103
real10kg_smc_steady         startup_0_3s 0.234106 0.231298 0.036077        0.023816                 0.000936
real10kg_smc_steady       pre_step_3_10s 0.118355 0.117872 0.010637        0.008270                 0.000128
real10kg_smc_steady step_recovery_10_13s 0.085739 0.081698 0.023902        0.118431                 0.003516
real10kg_smc_steady   post_step_13_20p9s 0.132774 0.131221 0.019924        0.024477                 0.000101
```

## Figures
- No figures generated.

## Files
- raw_metrics.csv
- summary_by_controller_scenario.csv
- overall_summary.csv
- window_metrics.csv
- window_summary.csv
- timeseries/timeseries_3d.csv
