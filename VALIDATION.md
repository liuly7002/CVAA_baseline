# Validation status

The final refactor was checked at source level and with dependency-light tests.

## Completed checks

- All project Python files parse under Python 3.8 grammar.
- All project Python files compile successfully in the build environment.
- `python -m unittest discover -s tests` passes 4 tests.
- Refactored actor projection/matching functions were compared against the
  development-stage `actor_instance_matcher.py` on synthetic vehicle/pedestrian
  examples and produced identical projected boxes.
- Refactored adaptive mask, local crop, and seam-mask helpers were compared
  against the development-stage counterfactual generator and produced identical
  outputs on synthetic masks.
- OpenCV fallback inpainting verifies
  `outside_mask_changed_pixels == 0`.
- Refactored AD/FD code reproduces the previously observed smoke-test results:
  - actor 3706: AD 0.011142875381546338, FD 0.0069580078125
  - actor 3707: AD 0.02690848117396292, FD 0.0009765625

## Not executed in this build sandbox

A complete FLUX + LaMa + original SimLingo end-to-end run was not executed here
because the user-local model weights, original SimLingo environment, and CARLA
route dataset are not present in this sandbox.

Before publishing paper-scale results, run the README smoke-test configuration
on the same route previously used during development and inspect the generated
waypoint debug figures.
