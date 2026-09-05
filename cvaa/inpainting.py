from __future__ import annotations

import gc
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image


GENERIC_REMOVAL_PROMPT = (
    "Photorealistic front-facing autonomous-driving camera image. "
    "Remove the masked object completely. "
    "Reconstruct the empty background that would naturally be visible behind the removed object. "
    "Continue the road surface, lane markings, curb, sidewalk, vegetation, buildings, "
    "guardrails, terrain, shadows, and distant background consistently with the surrounding scene. "
    "The masked region must not contain a replacement traffic participant, duplicated object, "
    "text, logo, or artificial artifact."
)

CLASS_REMOVAL_PROMPTS = {
    "vehicle": (
        "Remove the masked vehicle completely. "
        "The vehicle no longer exists in the scene. "
        "Reconstruct the empty road and background that would be visible behind it. "
        "The masked region must contain no car, truck, bus, motorcycle, bicycle, pedestrian, "
        "vehicle-shaped object, duplicated traffic participant, or residual vehicle body. "
        "Continue road surface, lane markings, curb, vegetation, guardrails, buildings, "
        "terrain, shadows, and distant background naturally and photorealistically."
    ),
    "pedestrian": (
        "Remove the masked pedestrian completely. "
        "The pedestrian no longer exists in the scene. "
        "Reconstruct the empty road, sidewalk, curb, terrain, or background that would be visible behind them. "
        "The masked region must contain no person, rider, cyclist, vehicle, human-shaped object, "
        "duplicated traffic participant, or residual body. "
        "Preserve the surrounding road scene naturally and photorealistically."
    ),
    "traffic_cone": (
        "Remove the masked traffic cone completely. "
        "Reconstruct the empty road surface or background behind it. "
        "Do not generate a replacement cone, barrier, vehicle, pedestrian, or other traffic object."
    ),
    "traffic_warning": (
        "Remove the masked traffic-warning object completely. "
        "Reconstruct the empty road surface and background behind it. "
        "Do not generate a replacement sign, warning object, barrier, vehicle, or pedestrian."
    ),
    "barrier": (
        "Remove the masked barrier completely. "
        "Reconstruct the empty road, shoulder, curb, terrain, or background behind it. "
        "Do not generate a replacement barrier, vehicle, pedestrian, or other traffic object."
    ),
}


def removal_prompt_for_class(
    actor_class: str,
    override: Optional[str] = None,
) -> str:
    if override:
        return str(override)
    actor_class = str(actor_class or "").lower()
    return CLASS_REMOVAL_PROMPTS.get(
        actor_class,
        GENERIC_REMOVAL_PROMPT,
    )


def mask_bbox_xyxy(mask_u8: np.ndarray) -> List[int]:
    ys, xs = np.nonzero(mask_u8 > 0)
    if len(xs) == 0:
        raise ValueError("Mask is empty.")
    return [
        int(xs.min()),
        int(ys.min()),
        int(xs.max()),
        int(ys.max()),
    ]


def bbox_wh_xyxy(bbox_xyxy: List[int]) -> Tuple[int, int]:
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    return int(x2 - x1 + 1), int(y2 - y1 + 1)


def adaptive_expand_mask(
    exact_mask_u8: np.ndarray,
    ratio: float,
    min_px: int,
    max_px: int,
) -> Tuple[np.ndarray, int, List[int]]:
    ys, xs = np.nonzero(exact_mask_u8 > 0)
    if len(xs) == 0:
        raise ValueError("Exact actor mask is empty.")

    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()), int(ys.max())
    w = x2 - x1 + 1
    h = y2 - y1 + 1

    radius = int(round(float(ratio) * float(max(w, h))))
    radius = max(int(min_px), min(int(max_px), radius))

    if radius <= 0:
        expanded = exact_mask_u8.copy()
    else:
        k = 2 * radius + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (k, k),
        )
        expanded = cv2.dilate(
            exact_mask_u8,
            kernel,
            iterations=1,
        )

    return expanded, radius, [x1, y1, x2, y2]


def exact_composite(
    source: Image.Image,
    generated: Image.Image,
    intervention_mask_u8: np.ndarray,
) -> Image.Image:
    src = np.asarray(
        source.convert("RGB"),
        dtype=np.uint8,
    )
    gen = np.asarray(
        generated.convert("RGB"),
        dtype=np.uint8,
    )

    if src.shape != gen.shape:
        raise ValueError(
            "source/generated shape mismatch: %s vs %s"
            % (src.shape, gen.shape)
        )

    mask_bool = intervention_mask_u8 > 0
    out = src.copy()
    out[mask_bool] = gen[mask_bool]
    return Image.fromarray(out, mode="RGB")


def count_outside_mask_changed_pixels(
    source: Image.Image,
    counterfactual: Image.Image,
    intervention_mask_u8: np.ndarray,
) -> int:
    src = np.asarray(
        source.convert("RGB"),
        dtype=np.uint8,
    )
    cf = np.asarray(
        counterfactual.convert("RGB"),
        dtype=np.uint8,
    )
    outside = intervention_mask_u8 == 0
    changed = np.any(src != cf, axis=2) & outside
    return int(np.count_nonzero(changed))


def local_square_crop_xyxy(
    bbox_xyxy: List[int],
    image_w: int,
    image_h: int,
    context_scale: float,
    min_side_px: int,
    max_side_px: int,
) -> List[int]:
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    bw = max(1, x2 - x1 + 1)
    bh = max(1, y2 - y1 + 1)
    side = int(
        round(
            float(context_scale)
            * float(max(bw, bh))
        )
    )
    side = max(
        int(min_side_px),
        min(int(max_side_px), side),
    )

    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    left = int(round(cx - side / 2.0))
    top = int(round(cy - side / 2.0))
    right = left + side - 1
    bottom = top + side - 1

    if left < 0:
        right += -left
        left = 0
    if top < 0:
        bottom += -top
        top = 0
    if right >= image_w:
        shift = right - image_w + 1
        left -= shift
        right = image_w - 1
    if bottom >= image_h:
        shift = bottom - image_h + 1
        top -= shift
        bottom = image_h - 1

    left = max(0, left)
    top = max(0, top)
    right = min(image_w - 1, right)
    bottom = min(image_h - 1, bottom)

    if right <= left or bottom <= top:
        raise ValueError("Invalid local crop.")

    return [
        int(left),
        int(top),
        int(right),
        int(bottom),
    ]


def crop_image_and_mask(
    source: Image.Image,
    full_mask_u8: np.ndarray,
    crop_xyxy: List[int],
) -> Tuple[Image.Image, np.ndarray]:
    x1, y1, x2, y2 = [int(v) for v in crop_xyxy]

    src_np = np.asarray(
        source.convert("RGB"),
        dtype=np.uint8,
    )
    crop_img = src_np[
        y1 : y2 + 1,
        x1 : x2 + 1,
    ]
    crop_mask = full_mask_u8[
        y1 : y2 + 1,
        x1 : x2 + 1,
    ]

    if crop_img.size == 0 or crop_mask.size == 0:
        raise ValueError("Empty crop produced.")

    return Image.fromarray(
        crop_img,
        mode="RGB",
    ), crop_mask


def resize_crop_for_inpainting(
    crop_source: Image.Image,
    crop_mask_u8: np.ndarray,
    target_size: int,
) -> Tuple[Image.Image, Image.Image]:
    if int(target_size) <= 0:
        raise ValueError(
            "crop_target_size must be positive."
        )

    source_resized = crop_source.resize(
        (int(target_size), int(target_size)),
        resample=Image.Resampling.BICUBIC,
    )
    mask_pil = Image.fromarray(
        crop_mask_u8,
        mode="L",
    )
    mask_resized = mask_pil.resize(
        (int(target_size), int(target_size)),
        resample=Image.Resampling.NEAREST,
    ).convert("L")

    mask_np = np.asarray(
        mask_resized,
        dtype=np.uint8,
    )
    mask_np = np.where(
        mask_np > 127,
        255,
        0,
    ).astype(np.uint8)

    return source_resized, Image.fromarray(
        mask_np,
        mode="L",
    )


def paste_generated_crop_to_full(
    source: Image.Image,
    generated_crop: Image.Image,
    crop_xyxy: List[int],
) -> Image.Image:
    x1, y1, x2, y2 = [int(v) for v in crop_xyxy]
    out = source.copy()
    w = int(x2 - x1 + 1)
    h = int(y2 - y1 + 1)

    crop_back = generated_crop.resize(
        (w, h),
        resample=Image.Resampling.LANCZOS,
    )
    out.paste(
        crop_back,
        (x1, y1),
    )
    return out


def inner_seam_mask(
    mask_u8: np.ndarray,
    width_px: int,
) -> np.ndarray:
    binary = np.where(
        mask_u8 > 0,
        255,
        0,
    ).astype(np.uint8)

    width_px = int(max(0, width_px))
    if width_px <= 0:
        return binary

    k = 2 * width_px + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (k, k),
    )
    eroded = cv2.erode(
        binary,
        kernel,
        iterations=1,
    )
    ring = cv2.subtract(
        binary,
        eroded,
    )

    if int(np.count_nonzero(ring)) == 0:
        return binary
    return ring


# ---------------------------------------------------------------------------
# Minimal LaMa TorchScript compatibility layer.
#
# The preprocessing and TorchScript invocation mirror the public
# enesmsahin/simple-lama-inpainting implementation (Apache-2.0):
# https://github.com/enesmsahin/simple-lama-inpainting
#
# Keeping this small wrapper in-project avoids requiring that package's current
# Python version while preserving compatibility with big-lama.pt.
# ---------------------------------------------------------------------------

def _get_image_chw(
    image: Any,
) -> np.ndarray:
    if isinstance(image, Image.Image):
        img = np.array(image)
    elif isinstance(image, np.ndarray):
        img = image.copy()
    else:
        raise TypeError(
            "Input image must be PIL.Image or numpy array."
        )

    if img.ndim == 3:
        img = np.transpose(
            img,
            (2, 0, 1),
        )
    elif img.ndim == 2:
        img = img[np.newaxis, ...]

    if img.ndim != 3:
        raise ValueError(
            "Unexpected image shape: %s" % (img.shape,)
        )

    return img.astype(np.float32) / 255.0


def _ceil_modulo(
    value: int,
    modulo: int,
) -> int:
    if value % modulo == 0:
        return value
    return (
        value // modulo + 1
    ) * modulo


def _pad_img_to_modulo(
    img: np.ndarray,
    modulo: int,
) -> np.ndarray:
    _, height, width = img.shape
    out_height = _ceil_modulo(
        height,
        modulo,
    )
    out_width = _ceil_modulo(
        width,
        modulo,
    )

    return np.pad(
        img,
        (
            (0, 0),
            (0, out_height - height),
            (0, out_width - width),
        ),
        mode="symmetric",
    )


def _prepare_lama_inputs(
    image: Image.Image,
    mask: Image.Image,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    out_image = _get_image_chw(image)
    out_mask = _get_image_chw(mask)

    out_image = _pad_img_to_modulo(
        out_image,
        8,
    )
    out_mask = _pad_img_to_modulo(
        out_mask,
        8,
    )

    image_tensor = (
        torch.from_numpy(out_image)
        .unsqueeze(0)
        .to(device)
    )
    mask_tensor = (
        torch.from_numpy(out_mask)
        .unsqueeze(0)
        .to(device)
    )
    mask_tensor = (
        mask_tensor > 0
    ) * 1

    return image_tensor, mask_tensor


class LamaTorchscript(object):
    """
    Minimal in-project wrapper around big-lama.pt.

    This intentionally avoids the simple-lama-inpainting package so the final
    project can remain in the official SimLingo Python 3.8 environment.
    """

    def __init__(
        self,
        model_path: Path,
        device: str,
    ) -> None:
        self.model_path = (
            model_path.expanduser().resolve()
        )
        if not self.model_path.exists():
            raise FileNotFoundError(
                "Missing LaMa model: %s"
                % self.model_path
            )

        self.device = torch.device(
            str(device)
        )
        lama_load_start = time.perf_counter()
        print(
            "[LaMa] loading %s on %s"
            % (
                self.model_path,
                self.device,
            )
        )

        self.model = torch.jit.load(
            str(self.model_path),
            map_location=self.device,
        )
        self.model.eval()
        self.model.to(self.device)
        self.load_seconds = (
            time.perf_counter()
            - lama_load_start
        )
        print(
            "[LaMa] ready (load %.3fs)"
            % self.load_seconds
        )

    def __call__(
        self,
        image: Image.Image,
        mask: Image.Image,
    ) -> Image.Image:
        source_size = image.size
        image_tensor, mask_tensor = (
            _prepare_lama_inputs(
                image,
                mask,
                self.device,
            )
        )

        with torch.inference_mode():
            inpainted = self.model(
                image_tensor,
                mask_tensor,
            )

        cur = (
            inpainted[0]
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
        )
        cur = np.clip(
            cur * 255,
            0,
            255,
        ).astype(np.uint8)

        # Remove any modulo-8 padding.
        cur = cur[
            : source_size[1],
            : source_size[0],
        ]

        return Image.fromarray(
            cur,
            mode="RGB",
        )

    def close(self) -> None:
        if hasattr(self, "model"):
            del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def save_inpainting_diagnostic(
    source: Image.Image,
    mask: Image.Image,
    lama_generated: Image.Image,
    flux_generated: Image.Image,
    counterfactual: Image.Image,
    out_path: Path,
) -> None:
    w, h = source.size
    mask_rgb = Image.merge(
        "RGB",
        (mask, mask, mask),
    )

    canvas = Image.new(
        "RGB",
        (w * 5, h),
    )
    canvas.paste(source, (0, 0))
    canvas.paste(mask_rgb, (w, 0))
    canvas.paste(
        lama_generated,
        (w * 2, 0),
    )
    canvas.paste(
        flux_generated,
        (w * 3, 0),
    )
    canvas.paste(
        counterfactual,
        (w * 4, 0),
    )

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    canvas.save(
        out_path,
        format="JPEG",
        quality=95,
    )


class InpaintingEngine(object):
    def __init__(
        self,
        cfg: Dict[str, Any],
        lama_model_path: Path,
        flux_model: str,
    ) -> None:
        engine_load_start = time.perf_counter()

        self.cfg = cfg
        self.inpaint_cfg = cfg["inpainting"]
        self.mask_cfg = cfg["mask"]
        self.debug_cfg = cfg["debug"]
        self.output_cfg = cfg["output"]

        self.backend = str(
            self.inpaint_cfg["backend"]
        )
        self.lama = None
        self.pipe = None
        self.torch_module = torch

        # 性能统计只记录 wall-clock，不改变任何模型行为。
        self.lama_load_seconds = 0.0
        self.flux_load_seconds = 0.0
        self.engine_load_seconds = 0.0

        if self.backend in (
            "flux_fill",
            "lama_only",
        ):
            self.lama = LamaTorchscript(
                model_path=lama_model_path,
                device=str(
                    self.inpaint_cfg[
                        "lama_device"
                    ]
                ),
            )
            self.lama_load_seconds = float(
                getattr(
                    self.lama,
                    "load_seconds",
                    0.0,
                )
            )

        if (
            self.backend == "flux_fill"
            and str(
                self.inpaint_cfg[
                    "flux_refine_mode"
                ]
            )
            != "none"
        ):
            self._load_flux(flux_model)

        self.engine_load_seconds = (
            time.perf_counter()
            - engine_load_start
        )

    def _load_flux(
        self,
        flux_model: str,
    ) -> None:
        flux_load_start = time.perf_counter()

        try:
            from diffusers import (
                FluxFillPipeline,
            )
        except ImportError as e:
            raise RuntimeError(
                "FluxFillPipeline is unavailable in the cvaa_fill worker. "
                "请检查 config.yaml 中 environments.cvaa_fill_*，并在 "
                "cvaa_fill 环境中安装 diffusers==0.32.2。"
                "不要把 FLUX/diffusers 依赖安装到 simlingo 环境。"
            ) from e

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        dtype_name = str(
            self.inpaint_cfg["dtype"]
        )
        if dtype_name not in dtype_map:
            raise ValueError(
                "Unsupported FLUX dtype: %s"
                % dtype_name
            )

        print(
            "[FLUX] loading %s"
            % flux_model
        )
        self.pipe = (
            FluxFillPipeline.from_pretrained(
                flux_model,
                torch_dtype=dtype_map[
                    dtype_name
                ],
            )
        )

        if bool(
            self.inpaint_cfg[
                "sequential_cpu_offload"
            ]
        ):
            self.pipe.enable_sequential_cpu_offload()
            print(
                "[FLUX] sequential CPU offload enabled"
            )
        elif bool(
            self.inpaint_cfg[
                "cpu_offload"
            ]
        ):
            self.pipe.enable_model_cpu_offload()
            print(
                "[FLUX] model CPU offload enabled"
            )
        else:
            self.pipe = self.pipe.to(
                str(
                    self.inpaint_cfg[
                        "device"
                    ]
                )
            )

        if bool(
            self.inpaint_cfg[
                "vae_tiling"
            ]
        ):
            self.pipe.vae.enable_tiling()
            print(
                "[FLUX] VAE tiling enabled"
            )

        self.pipe.set_progress_bar_config(
            disable=False
        )

        self.flux_load_seconds = (
            time.perf_counter()
            - flux_load_start
        )
        print(
            "[FLUX] ready (load %.3fs)"
            % self.flux_load_seconds
        )

    def _run_flux(
        self,
        source: Image.Image,
        mask: Image.Image,
        prompt: str,
    ) -> Image.Image:
        if self.pipe is None:
            return source

        if bool(
            self.inpaint_cfg[
                "cpu_offload"
            ]
        ) or bool(
            self.inpaint_cfg[
                "sequential_cpu_offload"
            ]
        ):
            generator_device = "cpu"
        elif (
            str(
                self.inpaint_cfg["device"]
            ).startswith("cuda")
            and torch.cuda.is_available()
        ):
            generator_device = "cuda"
        else:
            generator_device = "cpu"

        generator = torch.Generator(
            device=generator_device
        )
        generator.manual_seed(
            int(
                self.inpaint_cfg["seed"]
            )
        )

        width, height = source.size
        with torch.inference_mode():
            output = self.pipe(
                prompt=prompt,
                image=source,
                mask_image=mask,
                width=width,
                height=height,
                num_inference_steps=int(
                    self.inpaint_cfg[
                        "num_inference_steps"
                    ]
                ),
                guidance_scale=float(
                    self.inpaint_cfg[
                        "guidance_scale"
                    ]
                ),
                max_sequence_length=int(
                    self.inpaint_cfg[
                        "max_sequence_length"
                    ]
                ),
                generator=generator,
            ).images[0]

        output = output.convert("RGB")
        if output.size != source.size:
            output = output.resize(
                source.size,
                resample=Image.Resampling.LANCZOS,
            )
        return output

    def _run_opencv(
        self,
        source: Image.Image,
        mask_np: np.ndarray,
    ) -> Image.Image:
        rgb = np.asarray(
            source,
            dtype=np.uint8,
        )
        bgr = cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2BGR,
        )

        method = (
            cv2.INPAINT_TELEA
            if str(
                self.inpaint_cfg[
                    "opencv_method"
                ]
            )
            == "telea"
            else cv2.INPAINT_NS
        )

        out_bgr = cv2.inpaint(
            bgr,
            mask_np,
            float(
                self.inpaint_cfg[
                    "opencv_radius"
                ]
            ),
            method,
        )
        out_rgb = cv2.cvtColor(
            out_bgr,
            cv2.COLOR_BGR2RGB,
        )
        return Image.fromarray(
            out_rgb,
            mode="RGB",
        )

    def generate(
        self,
        source_path: Path,
        actor_class: str,
        exact_mask_u8: np.ndarray,
        output_path: Path,
        diagnostic_path: Optional[Path] = None,
        save_exact_mask_path: Optional[Path] = None,
        save_intervention_mask_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        total_start = time.perf_counter()

        source_load_start = time.perf_counter()
        source = Image.open(
            source_path
        ).convert("RGB")
        source_load_seconds = (
            time.perf_counter()
            - source_load_start
        )

        preprocess_start = time.perf_counter()

        exact_pixels = int(
            np.count_nonzero(
                exact_mask_u8
            )
        )
        if exact_pixels < int(
            self.mask_cfg[
                "min_mask_pixels"
            ]
        ):
            raise SkipIntervention(
                "exact mask pixels %d < min_mask_pixels %d"
                % (
                    exact_pixels,
                    int(
                        self.mask_cfg[
                            "min_mask_pixels"
                        ]
                    ),
                )
            )

        exact_bbox = mask_bbox_xyxy(
            exact_mask_u8
        )
        exact_w, exact_h = (
            bbox_wh_xyxy(
                exact_bbox
            )
        )

        if min(
            exact_w,
            exact_h,
        ) < int(
            self.mask_cfg[
                "min_object_short_side_px"
            ]
        ):
            raise SkipIntervention(
                "exact bbox short side %d < %d"
                % (
                    min(
                        exact_w,
                        exact_h,
                    ),
                    int(
                        self.mask_cfg[
                            "min_object_short_side_px"
                        ]
                    ),
                )
            )

        if exact_pixels < int(
            self.mask_cfg[
                "min_exact_mask_pixels"
            ]
        ):
            raise SkipIntervention(
                "exact mask pixels %d < min_exact_mask_pixels %d"
                % (
                    exact_pixels,
                    int(
                        self.mask_cfg[
                            "min_exact_mask_pixels"
                        ]
                    ),
                )
            )

        mask_np, dilation_radius, exact_bbox = (
            adaptive_expand_mask(
                exact_mask_u8=exact_mask_u8,
                ratio=float(
                    self.mask_cfg[
                        "adaptive_dilate_ratio"
                    ]
                ),
                min_px=int(
                    self.mask_cfg[
                        "adaptive_dilate_min_px"
                    ]
                ),
                max_px=int(
                    self.mask_cfg[
                        "adaptive_dilate_max_px"
                    ]
                ),
            )
        )

        if source.size != (
            mask_np.shape[1],
            mask_np.shape[0],
        ):
            raise ValueError(
                "source/mask shape mismatch: %s vs %s"
                % (
                    source.size,
                    mask_np.shape[::-1],
                )
            )

        if save_exact_mask_path is not None:
            save_exact_mask_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            cv2.imwrite(
                str(
                    save_exact_mask_path
                ),
                exact_mask_u8,
            )

        if (
            save_intervention_mask_path
            is not None
        ):
            save_intervention_mask_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            cv2.imwrite(
                str(
                    save_intervention_mask_path
                ),
                mask_np,
            )

        mask = Image.fromarray(
            mask_np,
            mode="L",
        )

        local_crop_enabled = bool(
            self.inpaint_cfg[
                "local_crop_enabled"
            ]
        )

        if local_crop_enabled:
            crop_xyxy = (
                local_square_crop_xyxy(
                    bbox_xyxy=exact_bbox,
                    image_w=source.size[0],
                    image_h=source.size[1],
                    context_scale=float(
                        self.inpaint_cfg[
                            "crop_context_scale"
                        ]
                    ),
                    min_side_px=int(
                        self.inpaint_cfg[
                            "crop_min_side_px"
                        ]
                    ),
                    max_side_px=int(
                        self.inpaint_cfg[
                            "crop_max_side_px"
                        ]
                    ),
                )
            )
            (
                crop_source,
                crop_mask_np,
            ) = crop_image_and_mask(
                source,
                mask_np,
                crop_xyxy,
            )
            (
                inpaint_source,
                inpaint_mask,
            ) = resize_crop_for_inpainting(
                crop_source,
                crop_mask_np,
                int(
                    self.inpaint_cfg[
                        "crop_target_size"
                    ]
                ),
            )
        else:
            crop_xyxy = [
                0,
                0,
                source.size[0] - 1,
                source.size[1] - 1,
            ]
            crop_source = source
            crop_mask_np = mask_np
            inpaint_source = source
            inpaint_mask = mask

        preprocess_seconds = (
            time.perf_counter()
            - preprocess_start
        )

        lama_start = time.perf_counter()
        if self.backend in (
            "flux_fill",
            "lama_only",
        ):
            if self.lama is None:
                raise RuntimeError(
                    "LaMa was not initialized."
                )
            lama_local = self.lama(
                inpaint_source,
                inpaint_mask,
            )
        else:
            lama_local = self._run_opencv(
                inpaint_source,
                np.asarray(
                    inpaint_mask,
                    dtype=np.uint8,
                ),
            )

        lama_seconds = (
            time.perf_counter()
            - lama_start
        )

        refine_mode = str(
            self.inpaint_cfg[
                "flux_refine_mode"
            ]
        )

        flux_seconds = 0.0

        if (
            self.backend == "flux_fill"
            and refine_mode != "none"
        ):
            if refine_mode == "full":
                flux_mask_local = (
                    inpaint_mask
                )
            else:
                if local_crop_enabled:
                    seam_crop_np = (
                        inner_seam_mask(
                            crop_mask_np,
                            int(
                                self.inpaint_cfg[
                                    "flux_seam_width_px"
                                ]
                            ),
                        )
                    )
                    _, flux_mask_local = (
                        resize_crop_for_inpainting(
                            crop_source,
                            seam_crop_np,
                            int(
                                self.inpaint_cfg[
                                    "crop_target_size"
                                ]
                            ),
                        )
                    )
                else:
                    seam_full_np = (
                        inner_seam_mask(
                            mask_np,
                            int(
                                self.inpaint_cfg[
                                    "flux_seam_width_px"
                                ]
                            ),
                        )
                    )
                    flux_mask_local = (
                        Image.fromarray(
                            seam_full_np,
                            mode="L",
                        )
                    )

            prompt = removal_prompt_for_class(
                actor_class,
                self.inpaint_cfg.get(
                    "prompt_override"
                ),
            )
            flux_start = time.perf_counter()
            flux_local = self._run_flux(
                lama_local,
                flux_mask_local,
                prompt,
            )
            flux_seconds = (
                time.perf_counter()
                - flux_start
            )
        else:
            prompt = removal_prompt_for_class(
                actor_class,
                self.inpaint_cfg.get(
                    "prompt_override"
                ),
            )
            flux_local = lama_local

        postprocess_start = time.perf_counter()

        if local_crop_enabled:
            lama_full = (
                paste_generated_crop_to_full(
                    source,
                    lama_local,
                    crop_xyxy,
                )
            )
            flux_full = (
                paste_generated_crop_to_full(
                    source,
                    flux_local,
                    crop_xyxy,
                )
            )
        else:
            lama_full = lama_local
            flux_full = flux_local

        counterfactual = exact_composite(
            source,
            flux_full,
            mask_np,
        )

        outside_changed = (
            count_outside_mask_changed_pixels(
                source,
                counterfactual,
                mask_np,
            )
        )
        if outside_changed != 0:
            raise RuntimeError(
                "CVAA invariant violated: "
                "%d outside-mask pixels changed"
                % outside_changed
            )

        postprocess_seconds = (
            time.perf_counter()
            - postprocess_start
        )

        output_save_start = time.perf_counter()
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        counterfactual.save(
            output_path,
            format="PNG",
            optimize=True,
        )
        output_save_seconds = (
            time.perf_counter()
            - output_save_start
        )

        debug_save_seconds = 0.0
        if diagnostic_path is not None:
            debug_save_start = time.perf_counter()
            save_inpainting_diagnostic(
                source=source,
                mask=mask,
                lama_generated=lama_full,
                flux_generated=flux_full,
                counterfactual=counterfactual,
                out_path=diagnostic_path,
            )
            debug_save_seconds = (
                time.perf_counter()
                - debug_save_start
            )

        total_seconds = (
            time.perf_counter()
            - total_start
        )

        return {
            "exact_mask_pixels": exact_pixels,
            "mask_pixels_used": int(
                np.count_nonzero(
                    mask_np
                )
            ),
            "exact_bbox_xyxy": exact_bbox,
            "adaptive_dilation_radius_px": int(
                dilation_radius
            ),
            "local_crop_enabled": (
                local_crop_enabled
            ),
            "crop_xyxy": crop_xyxy,
            "crop_target_size": (
                int(
                    self.inpaint_cfg[
                        "crop_target_size"
                    ]
                )
                if local_crop_enabled
                else None
            ),
            "backend": self.backend,
            "pipeline_strategy": (
                "lama_then_flux"
                if self.backend
                == "flux_fill"
                else "lama_only"
                if self.backend
                == "lama_only"
                else "opencv_debug"
            ),
            "flux_refine_mode": (
                refine_mode
                if self.backend
                == "flux_fill"
                else None
            ),
            "outside_mask_changed_pixels": (
                outside_changed
            ),
            "prompt": prompt,
            "performance": {
                "total_seconds": float(
                    total_seconds
                ),
                "source_image_load_seconds": float(
                    source_load_seconds
                ),
                "preprocess_seconds": float(
                    preprocess_seconds
                ),
                "lama_seconds": float(
                    lama_seconds
                ),
                "flux_seconds": float(
                    flux_seconds
                ),
                "postprocess_seconds": float(
                    postprocess_seconds
                ),
                "output_save_seconds": float(
                    output_save_seconds
                ),
                "debug_save_seconds": float(
                    debug_save_seconds
                ),
            },
        }

    def close(self) -> None:
        if self.pipe is not None:
            try:
                del self.pipe
            except Exception:
                pass
            self.pipe = None

        if self.lama is not None:
            try:
                self.lama.close()
            except Exception:
                pass
            self.lama = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class SkipIntervention(RuntimeError):
    pass
