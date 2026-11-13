import json
import time

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp


class MuralDamageSegmenter:

    def __init__(
        self,
        checkpoint_path,
        metadata_path=None
    ):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device
        )

        metadata = {}

        if metadata_path:
            try:
                with open(
                    metadata_path,
                    "r",
                    encoding="utf-8"
                ) as file:
                    metadata = json.load(file)
            except FileNotFoundError:
                metadata = {}

        self.encoder_name = checkpoint.get(
            "encoder_name",
            metadata.get("encoder_name", "efficientnet-b3")
        )

        self.image_size = int(
            checkpoint.get(
                "image_size",
                metadata.get("image_size", 384)
            )
        )

        self.threshold = float(
            checkpoint.get(
                "threshold",
                metadata.get("threshold", 0.5)
            )
        )

        self.mean = np.asarray(
            checkpoint.get(
                "mean",
                metadata.get(
                    "mean",
                    [0.485, 0.456, 0.406]
                )
            ),
            dtype=np.float32
        )

        self.std = np.asarray(
            checkpoint.get(
                "std",
                metadata.get(
                    "std",
                    [0.229, 0.224, 0.225]
                )
            ),
            dtype=np.float32
        )

        self.model = smp.UnetPlusPlus(
            encoder_name=self.encoder_name,
            encoder_weights=None,
            in_channels=int(
                checkpoint.get("in_channels", 3)
            ),
            classes=int(
                checkpoint.get("classes", 1)
            ),
            activation=None
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.to(self.device)
        self.model.eval()

        print(
            f"Segmentation model loaded on {self.device}: "
            f"UnetPlusPlus/{self.encoder_name}"
        )

    def preprocess(self, image_rgb):

        resized = cv2.resize(
            image_rgb,
            (
                self.image_size,
                self.image_size
            ),
            interpolation=cv2.INTER_AREA
        )

        x = (
            resized.astype(np.float32)
            / 255.0
        )

        x = (
            x
            -
            self.mean
        ) / self.std

        x = np.transpose(
            x,
            (
                2,
                0,
                1
            )
        )

        tensor = torch.from_numpy(
            x
        ).unsqueeze(0).float().to(
            self.device
        )

        return tensor

    @staticmethod
    def cleanup_mask(
        probability,
        threshold,
        min_area=15
    ):
        binary = (
            probability >= threshold
        ).astype(np.uint8) * 255

        kernel = np.ones(
            (3, 3),
            dtype=np.uint8
        )

        # Connect tiny gaps in damaged regions.
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=1
        )

        count, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                binary,
                connectivity=8
            )
        )

        cleaned = np.zeros_like(binary)

        for component in range(1, count):
            area = stats[
                component,
                cv2.CC_STAT_AREA
            ]

            if area >= min_area:
                cleaned[
                    labels == component
                ] = 255

        return cleaned

    @torch.no_grad()
    def predict(self, image_rgb):

        original_h, original_w = image_rgb.shape[:2]

        tensor = self.preprocess(
            image_rgb
        )

        logits = self.model(
            tensor
        )

        probability = torch.sigmoid(
            logits
        )[0, 0].detach().cpu().numpy()

        clean_mask = self.cleanup_mask(
            probability,
            self.threshold,
            min_area=15
        )

        mask_original = cv2.resize(
            clean_mask,
            (
                original_w,
                original_h
            ),
            interpolation=cv2.INTER_NEAREST
        )

        probability_original = cv2.resize(
            probability,
            (
                original_w,
                original_h
            ),
            interpolation=cv2.INTER_LINEAR
        )

        return {
            "probability": probability_original,
            "mask": mask_original,
            "threshold": self.threshold,
            "probability_min": float(
                probability.min()
            ),
            "probability_max": float(
                probability.max()
            ),
            "probability_mean": float(
                probability.mean()
            )
        }


class CombinedDamageSegmenter:
    """Retain every damaged pixel detected by either independent model."""

    def __init__(self, checkpoint_path, metadata_path=None, scratch_model_dir=None):
        from .scratch_segmentation import ScratchSegmenter

        self.trained = MuralDamageSegmenter(checkpoint_path, metadata_path)
        self.scratches = ScratchSegmenter(scratch_model_dir)

    @staticmethod
    def combine_masks(trained_mask, scratch_mask):
        if trained_mask.ndim != 2 or trained_mask.shape != scratch_mask.shape:
            raise ValueError("Both segmentation masks must have matching image dimensions.")
        return ((trained_mask > 0) | (scratch_mask > 0)).astype(np.uint8) * 255

    def predict(self, image_rgb):
        started = time.perf_counter()
        trained = self.trained.predict(image_rgb)
        trained_seconds = time.perf_counter() - started
        started = time.perf_counter()
        scratches = self.scratches.predict(image_rgb)
        scratch_seconds = time.perf_counter() - started
        trained_mask, scratch_mask = trained["mask"], scratches["mask"]
        if trained_mask.shape != image_rgb.shape[:2] or scratch_mask.shape != image_rgb.shape[:2]:
            raise ValueError("Segmentation masks must align with the original image.")
        combined = self.combine_masks(trained_mask, scratch_mask)
        # Existing probability/threshold fields remain explicitly U-Net++'s;
        # probabilities from different models are not treated as calibrated.
        return {
            **trained,
            "mask": combined,
            "trained_mask": trained_mask,
            "scratch_mask": scratch_mask,
            "detection": {
                "method": "U-Net++ + ZeroScratches (union)",
                "trained_percentage": round(float(np.mean(trained_mask > 0) * 100), 2),
                "scratch_percentage": round(float(np.mean(scratch_mask > 0) * 100), 2),
                "additional_percentage": round(float(np.mean(
                    (scratch_mask > 0) & (trained_mask == 0)
                ) * 100), 2),
                "scratch_threshold": scratches["threshold"],
                "scratch_probability": {
                    key: round(scratches[f"probability_{key}"], 4)
                    for key in ("min", "mean", "max")
                },
            },
            "detector_timings": {
                "trained_segmentation": round(trained_seconds, 2),
                "scratch_detection": round(scratch_seconds, 2),
            },
        }
