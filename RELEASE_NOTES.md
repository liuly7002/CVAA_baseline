# Release notes — final batch version

This release replaces the development-stage multi-script workflow with a single
configuration-driven batch pipeline.

## Main changes

- Removed command-line configuration for paths, debug switches, model settings,
  thresholds, and output policy. Everything is centralized in `config.yaml`.
- Replaced persistent Stage-1/2/3 intermediate directories with in-memory
  matching/masks and bounded temporary counterfactual chunks.
- Added explicit model scheduling: LaMa/FLUX are released before original
  SimLingo is loaded, reducing peak GPU memory usage.
- Added automatic deletion of temporary counterfactual PNGs after each chunk.
- Kept optional persistent masks/counterfactuals only behind `output.*` switches.
- Integrated actor matching, mask construction, object removal, original
  SimLingo paired inference, AD/FD calculation, and actor ranking.
- Integrated the waypoint debug view:
  - red: predicted `pred_speed_wps`
  - green: ground-truth future ego waypoints
  - original and counterfactual images shown side by side.
- Added per-route resume/rebuild behavior and global batch summaries.
- Added strict original-SimLingo source-tree checking.
- Removed the `simple-lama-inpainting` package dependency by using the public
  `big-lama.pt` TorchScript model directly, allowing the project to stay on the
  official SimLingo Python 3.8 environment.
