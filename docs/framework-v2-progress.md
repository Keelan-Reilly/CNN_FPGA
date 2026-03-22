# Framework V2 Progress

- Preserved the existing measured CNN accelerator study and its aggregate/plot pipeline.
- Added an explicit MAC-array evaluation layer with workload classes, adaptive switching, break-even analysis, and policy outputs.
- Promoted prior MAC-array static evidence into a first-class evidence registry with provenance ids and source metadata rather than leaving it embedded in the top-level config.
- Standardized generated v2 outputs under `results/fpga/framework_v2/`.
