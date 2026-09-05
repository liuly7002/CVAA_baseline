# Third-party notices

This project is an experimental CVAA-style baseline implementation built around
CARLA data and the **original SimLingo** model.

## Original SimLingo

- Repository: https://github.com/RenzKa/simlingo
- The pipeline imports the original SimLingo source tree at runtime and loads its
  released checkpoint. SimLingo source code is **not vendored** into this project.
- Users are responsible for following the license and model terms distributed by
  the SimLingo authors.

## LaMa / simple-lama-inpainting compatibility wrapper

`cvaa/inpainting.py` contains a minimal TorchScript wrapper and image/mask
pre-processing logic compatible with the public `big-lama.pt` workflow.

The wrapper behavior was derived from the open-source project:

- https://github.com/enesmsahin/simple-lama-inpainting
- License: Apache-2.0
- Original LaMa: https://github.com/advimman/lama

The final project does **not** require the `simple-lama-inpainting` Python
package. This is intentional so the pipeline can remain in the original
SimLingo Python 3.8 environment.

## FLUX.1-Fill-dev

- Model: `black-forest-labs/FLUX.1-Fill-dev`
- Loaded at runtime through Hugging Face Diffusers.
- The model weights are not included in this repository. Users must accept and
  comply with the model's current distribution/license terms.

## Diffusers

- https://github.com/huggingface/diffusers
- This project recommends `diffusers==0.32.2` because it contains
  `FluxFillPipeline` while remaining compatible with Python 3.8.

## CARLA

CARLA simulator assets and APIs remain subject to CARLA's own license.
