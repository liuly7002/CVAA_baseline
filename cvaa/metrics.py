from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


def trajectory_distribution(
    value: Any,
    field_name: str,
) -> np.ndarray:
    """
    Normalize trajectory output into [K,T,2].
    """
    if value is None:
        raise ValueError("%s is None" % field_name)

    arr = np.asarray(
        value,
        dtype=np.float64,
    )
    if arr.size == 0:
        raise ValueError("%s is empty" % field_name)

    while (
        arr.ndim > 3
        and arr.shape[0] == 1
    ):
        arr = arr[0]

    if arr.ndim == 2:
        if arr.shape[-1] < 2:
            raise ValueError(
                "%s must end with XY coordinates; got %s"
                % (field_name, arr.shape)
            )
        arr = arr[:, :2][None, ...]

    elif arr.ndim == 3:
        if arr.shape[-1] < 2:
            raise ValueError(
                "%s must end with XY coordinates; got %s"
                % (field_name, arr.shape)
            )
        arr = arr[..., :2]

    else:
        raise ValueError(
            "%s: expected [T,2], [K,T,2], or batch-1 equivalent; got %s"
            % (field_name, arr.shape)
        )

    if not np.isfinite(arr).all():
        raise ValueError("%s contains NaN/Inf" % field_name)

    return arr


def mean_trajectory(
    value: Any,
    field_name: str,
) -> Tuple[np.ndarray, int]:
    dist = trajectory_distribution(
        value,
        field_name,
    )
    return dist.mean(axis=0), int(
        dist.shape[0]
    )


def compute_ad_fd_from_means(
    orig_mean: np.ndarray,
    cf_mean: np.ndarray,
) -> Tuple[float, float, np.ndarray]:
    if orig_mean.shape != cf_mean.shape:
        raise ValueError(
            "Original/counterfactual mean trajectory shape mismatch: %s vs %s"
            % (orig_mean.shape, cf_mean.shape)
        )

    if (
        orig_mean.ndim != 2
        or orig_mean.shape[1] != 2
    ):
        raise ValueError(
            "Mean trajectory must be [T,2], got %s"
            % (orig_mean.shape,)
        )

    delta = cf_mean - orig_mean
    per_timestep = np.linalg.norm(
        delta,
        axis=1,
    )

    ad = float(per_timestep.mean())
    fd = float(per_timestep[-1])
    return ad, fd, per_timestep


def compute_metric_pair(
    original_value: Any,
    counterfactual_value: Any,
    field_name: str,
) -> Dict[str, Any]:
    orig_mean, k_orig = mean_trajectory(
        original_value,
        "original.%s" % field_name,
    )
    cf_mean, k_cf = mean_trajectory(
        counterfactual_value,
        "counterfactual.%s" % field_name,
    )

    ad, fd, per_timestep = (
        compute_ad_fd_from_means(
            orig_mean,
            cf_mean,
        )
    )

    return {
        "AD": ad,
        "FD": fd,
        "T": int(orig_mean.shape[0]),
        "K_original": k_orig,
        "K_counterfactual": k_cf,
        "per_timestep_displacement": (
            per_timestep.tolist()
        ),
        "original_mean_trajectory": (
            orig_mean.tolist()
        ),
        "counterfactual_mean_trajectory": (
            cf_mean.tolist()
        ),
    }


def numeric_actor_sort_key(
    actor_id: Any,
) -> Tuple[int, Any]:
    text = str(actor_id)
    try:
        return 0, int(text)
    except Exception:
        return 1, text


def scene_stats(
    values: List[float],
) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "median": None,
        }

    arr = np.asarray(
        values,
        dtype=np.float64,
    )
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "median": float(np.median(arr)),
    }


def safe_top_zscore(
    values: List[float],
) -> Optional[float]:
    if len(values) < 2:
        return None

    arr = np.asarray(
        values,
        dtype=np.float64,
    )
    std = float(arr.std(ddof=0))
    if std <= 0.0:
        return None

    return float(
        (arr.max() - arr.mean()) / std
    )


def rank_actor_scores(
    actor_scores: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Rank within each (route, frame) by AD descending, FD descending tie-break.
    """
    grouped: Dict[
        Tuple[str, str],
        List[Dict[str, Any]],
    ] = defaultdict(list)

    for score in actor_scores:
        grouped[
            (
                str(score["route_id"]),
                str(score["frame"]),
            )
        ].append(score)

    frame_rankings: List[Dict[str, Any]] = []

    for (
        route_id,
        frame,
    ), scene_scores in sorted(
        grouped.items()
    ):
        ranked = sorted(
            scene_scores,
            key=lambda x: (
                -float(x["AD"]),
                -float(x["FD"]),
                numeric_actor_sort_key(
                    x["actor_id"]
                ),
            ),
        )

        for rank, score in enumerate(
            ranked,
            1,
        ):
            score["rank"] = rank

        ad_values = [
            float(s["AD"])
            for s in ranked
        ]
        fd_values = [
            float(s["FD"])
            for s in ranked
        ]

        frame_rankings.append(
            {
                "route_id": route_id,
                "frame": frame,
                "num_actors": len(ranked),
                "top_actor_id": ranked[0][
                    "actor_id"
                ],
                "top_actor_class": ranked[0][
                    "actor_class"
                ],
                "top_AD": ranked[0]["AD"],
                "top_FD": ranked[0]["FD"],
                "AD_stats": scene_stats(
                    ad_values
                ),
                "FD_stats": scene_stats(
                    fd_values
                ),
                "max_AD_zscore": (
                    safe_top_zscore(
                        ad_values
                    )
                ),
                "ranking": [
                    {
                        "rank": s["rank"],
                        "actor_id": s[
                            "actor_id"
                        ],
                        "actor_class": s[
                            "actor_class"
                        ],
                        "AD": s["AD"],
                        "FD": s["FD"],
                    }
                    for s in ranked
                ],
            }
        )

    return frame_rankings


def check_same_frame_original_consistency(
    frame_records: List[Dict[str, Any]],
    atol: float,
    rtol: float,
) -> None:
    """
    Input records must contain original_prediction.pred_route.
    """
    if len(frame_records) <= 1:
        return

    ref, _ = mean_trajectory(
        frame_records[0][
            "original_prediction"
        ]["pred_route"],
        "original.pred_route",
    )

    for record in frame_records[1:]:
        cur, _ = mean_trajectory(
            record[
                "original_prediction"
            ]["pred_route"],
            "original.pred_route",
        )

        if not (
            ref.shape == cur.shape
            and np.allclose(
                ref,
                cur,
                atol=atol,
                rtol=rtol,
            )
        ):
            raise RuntimeError(
                "Same-frame original pred_route differs across actor interventions: "
                "frame=%s actor=%s"
                % (
                    record["frame"],
                    record["actor_id"],
                )
            )
