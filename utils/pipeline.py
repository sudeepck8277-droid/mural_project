import os
import time

import cv2
import numpy as np

from .segmentation import (
    CombinedDamageSegmenter
)
from .restoration import (
    MuralRestorer
)
from .super_resolution import (
    MuralSuperResolver
)


class MuralRestorationPipeline:

    def __init__(
        self,
        segmentation_model_path,
        super_resolution_model_path,
        segmentation_metadata_path=None,
        restoration_model_path=None,
        scratch_model_dir=None
    ):
        required = {
            "segmentation model":
            segmentation_model_path,

            "super-resolution model":
            super_resolution_model_path
        }

        missing = [
            f"{name}: {path}"
            for name, path in required.items()
            if not os.path.exists(path)
        ]

        if missing:
            raise FileNotFoundError(
                "Missing required model files: "
                +
                " | ".join(missing)
            )

        self.segmenter = (
            CombinedDamageSegmenter(
                segmentation_model_path,
                segmentation_metadata_path,
                scratch_model_dir
            )
        )

        self.restorer = MuralRestorer(restoration_model_path)

        self.super_resolver = (
            MuralSuperResolver(
                super_resolution_model_path
            )
        )

    @staticmethod
    def create_overlay(
        image_rgb,
        mask
    ):
        overlay = image_rgb.copy()

        region = (
            mask > 127
        )

        if np.any(region):
            highlight = np.array(
                [255, 55, 55],
                dtype=np.float32
            )

            overlay[
                region
            ] = (
                0.55
                *
                overlay[
                    region
                ].astype(
                    np.float32
                )
                +
                0.45
                *
                highlight
            ).astype(
                np.uint8
            )

        return overlay

    def process(
        self,
        image_path,
        output_dir,
        run_id
    ):
        process_started = time.perf_counter()

        image_bgr = cv2.imread(
            image_path
        )

        if image_bgr is None:
            raise ValueError(
                "OpenCV could not read the uploaded image."
            )

        image_rgb = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB
        )

        original_h, original_w = (
            image_rgb.shape[:2]
        )

        # -------------------------
        # Stage 1: Segmentation
        # -------------------------
        stage_started = time.perf_counter()

        seg_result = (
            self.segmenter.predict(
                image_rgb
            )
        )

        segmentation_seconds = time.perf_counter() - stage_started

        damage_mask = seg_result[
            "mask"
        ]

        damage_percentage = float(
            np.mean(
                damage_mask > 127
            )
            *
            100.0
        )

        overlay_rgb = (
            self.create_overlay(
                image_rgb,
                damage_mask
            )
        )

        # -------------------------
        # Stage 2: Restoration
        # -------------------------
        stage_started = time.perf_counter()

        if damage_percentage > 0:
            restored_rgb = (
                self.restorer.restore(
                    image_rgb,
                    damage_mask
                )
            )
        else:
            restored_rgb = (
                image_rgb.copy()
            )

        restoration_seconds = time.perf_counter() - stage_started

        # -------------------------
        # Stage 3: ×4 EDSR
        # -------------------------
        stage_started = time.perf_counter()

        sr_rgb = (
            self.super_resolver.upscale(
                restored_rgb
            )
        )

        super_resolution_seconds = time.perf_counter() - stage_started

        os.makedirs(
            output_dir,
            exist_ok=True
        )

        mask_path = os.path.join(
            output_dir,
            f"{run_id}_damage_mask.png"
        )

        overlay_path = os.path.join(
            output_dir,
            f"{run_id}_damage_overlay.png"
        )

        restored_path = os.path.join(
            output_dir,
            f"{run_id}_restored.png"
        )

        sr_path = os.path.join(
            output_dir,
            f"{run_id}_super_resolved.png"
        )

        cv2.imwrite(
            mask_path,
            damage_mask
        )

        trained_mask_path = os.path.join(output_dir, f"{run_id}_trained_mask.png")
        scratch_mask_path = os.path.join(output_dir, f"{run_id}_scratch_mask.png")
        cv2.imwrite(trained_mask_path, seg_result["trained_mask"])
        cv2.imwrite(scratch_mask_path, seg_result["scratch_mask"])

        cv2.imwrite(
            overlay_path,
            cv2.cvtColor(
                overlay_rgb,
                cv2.COLOR_RGB2BGR
            )
        )

        cv2.imwrite(
            restored_path,
            cv2.cvtColor(
                restored_rgb,
                cv2.COLOR_RGB2BGR
            )
        )

        cv2.imwrite(
            sr_path,
            cv2.cvtColor(
                sr_rgb,
                cv2.COLOR_RGB2BGR
            )
        )

        sr_h, sr_w = sr_rgb.shape[:2]

        total_seconds = time.perf_counter() - process_started

        return {
            "restoration_method": self.restorer.name,
            "detection": seg_result["detection"],
            "trained_mask_path": trained_mask_path,
            "scratch_mask_path": scratch_mask_path,

            "mask_path":
            mask_path,

            "overlay_path":
            overlay_path,

            "restored_path":
            restored_path,

            "sr_path":
            sr_path,

            "damage_percentage":
            round(
                damage_percentage,
                2
            ),

            "threshold":
            round(
                float(
                    seg_result[
                        "threshold"
                    ]
                ),
                3
            ),

            "segmentation_probability": {
                "min": round(
                    seg_result[
                        "probability_min"
                    ],
                    4
                ),
                "max": round(
                    seg_result[
                        "probability_max"
                    ],
                    4
                ),
                "mean": round(
                    seg_result[
                        "probability_mean"
                    ],
                    4
                )
            },

            "timings": {
                **seg_result["detector_timings"],
                "segmentation": round(segmentation_seconds, 2),
                "restoration": round(restoration_seconds, 2),
                "super_resolution": round(super_resolution_seconds, 2),
                "total": round(total_seconds, 2)
            },

            "input_size":
            f"{original_w} × {original_h}",

            "sr_size":
            f"{sr_w} × {sr_h}"
        }
