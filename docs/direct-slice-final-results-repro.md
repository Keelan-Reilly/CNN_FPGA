# Direct-Slice Family Final Results Reproducibility

- Regenerate the full thesis-ready direct-slice family results pack with `make fpga_mac_direct_final_pack`.
- The target runs:
  - `python3 analysis/run_mac_array_direct_slice.py`
  - `python3 analysis/run_mac_array_framework.py --config experiments/configs/mac_array_framework_v2.json`
- The pipeline depends on the existing measured direct-slice aggregates already checked into `results/fpga/aggregates/`.

## Validated Domain

- Direct-slice family only: `baseline`, `shared_lut_saving`, `shared_dsp_reducing`.
- Directly measured grids: `4x4`, `8x4`, `8x8`.
- Validated local predictor and decision-surface domain: `16 <= mac_units <= 64` at `k_depth=32`.

## Evidence Boundary

- Directly measured:
  - The `4x4`, `8x4`, and `8x8` direct-slice lattice for `baseline`, `shared_lut_saving`, and `shared_dsp_reducing`.
  - All measured tradeoff, scaling, utility, and final architecture-role conclusions inside that lattice.
- Interpolated within measured domain:
  - The local direct-slice predictor for DSP, LUT, FF, latency, and throughput.
  - The bounded LUT-pressure and DSP-pressure decision boundaries inside the validated domain.
- Explicitly refused:
  - Any predictor-backed architecture choice outside `16 <= mac_units <= 64`.
  - Any smooth timing-sensitive decision surface.
- Caution-only:
  - WNS may be shown at measured points and in caution-only figures, but it is not promoted to trusted timing-sensitive interpolation.

## Thesis Mapping

- `final_tradeoff_figures/`
  - Results chapter figures showing what each architecture buys and costs.
- `final_predictor_validation_figures/`
  - Methods/validation figures showing which local predictors are exact, locally linear, or caution-only.
- `final_decision_surface_figures/`
  - Discussion figures showing when each architecture is worth choosing and where unsupported extrapolation begins.
- `final_design_rule_table.*`
  - Compact measured design-rule table for the main Results/Discussion subsection.
- `final_trust_calibration_table.*`
  - Compact trust/calibration table explaining measured truth, interpolation, and unsupported extrapolation.
- `final_architecture_choice_boundary_table.*`
  - Compact boundary table for the LUT-pressure, DSP-pressure, and baseline-default regions.
- `final_results_summary.*`
  - Concise thesis-style subsection text.
- `final_artifact_index.*`
  - Artifact-by-artifact index with purpose, trust status, and thesis-use note.

## Scope Guard

- These outputs are bounded to the isolated direct-slice family.
- They do not replace the broader shared-family framework model.
- They do not alter the framework policy engine.
