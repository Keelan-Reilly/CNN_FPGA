# Framework V2 Progress

- Preserved the existing measured CNN accelerator study and its aggregate/plot pipeline.
- Added an explicit MAC-array evaluation layer with workload classes, adaptive switching, break-even analysis, and policy outputs.
- Promoted prior MAC-array static evidence into a first-class evidence registry with provenance ids and source metadata rather than leaving it embedded in the top-level config.
- Standardized generated v2 outputs under `results/fpga/framework_v2/`.
- Extended the direct MAC-array slice from a calibrated baseline-only foothold into a three-scale, three-way directly measured bridge: 4x4, 8x4, and 8x8 baseline, `shared_lut_saving`, and `shared_dsp_reducing` on the isolated direct top.
- Added direct tradeoff artifacts under `results/fpga/framework_v2/direct_slice/` so measured baseline/shared evidence, lightweight model comparisons, and scope limits are visible in the same framework-facing path.
- Refined the taxonomy so framework `shared` outputs refer to the modelled shared family, while the direct 4x4, 8x4, and 8x8 shared slices are tracked as separate implementation-specific measured observations.
- Added a measured decision and scaling layer for the direct tradeoff so the repo now states when LUT-oriented sharing is useful, when DSP-oriented sharing is useful, and that the three-way role split survives from 4x4 to 8x4.
- Added a measured trust overlay so the repo now marks what is directly supported by measured implementation evidence, what is only directionally or partially supported at the shared-family level, and what remains extrapolated beyond measured support.
- Added a measured calibration layer so framework shared-family resource expectations are now read through direct-slice calibration aids and implementation-dependent caution bands instead of being presented as unconstrained numeric truths.
- Added a measured utility layer so the repo now states when the measured shared implementations are actually worth using: only when the relieved LUT or DSP bottleneck matters more than the measured throughput and latency penalty.
- Added a measured design-rule extraction layer so the repo now states the thesis-level conclusion directly: flexibility is only justified when the bottleneck it relieves matters more than the overhead it introduces.
