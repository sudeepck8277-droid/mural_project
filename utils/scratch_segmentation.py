"""ZeroScratches scratch detection aligned to the original mural coordinates."""

import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


class ScratchSegmenter:
    def __init__(self, model_dir=None, threshold=None, device=None):
        from zeroscratches.maskscratches import ScratchesDetector

        model_dir = Path(model_dir) if model_dir else (
            Path(__file__).resolve().parents[1] / "models" / "zeroscratches"
        )
        if not (model_dir / "detection" / "FT_Epoch_latest.pt").is_file():
            raise FileNotFoundError(
                "ZeroScratches detector weights are missing. "
                "Run python scripts/download_zeroscratches.py and restart Flask."
            )
        self.threshold = float(
            threshold if threshold is not None else
            os.environ.get("MURAL_SCRATCH_THRESHOLD", "0.4")
        )
        if not 0 < self.threshold < 1:
            raise ValueError("MURAL_SCRATCH_THRESHOLD must be between 0 and 1.")
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        # Use the library's detector and pretrained weights, not its eraser.
        # Direct tensor inference avoids its display-mask normalization and
        # allows padding/cropping without shifting the mask on odd image sizes.
        detector = ScratchesDetector(str(model_dir))
        self.model = detector.model_mask.to(self.device).eval()
        print(f"Scratch detector loaded on {self.device}: ZeroScratches")

    @torch.inference_mode()
    def predict(self, image_rgb):
        if (
            image_rgb.dtype != np.uint8 or image_rgb.ndim != 3
            or image_rgb.shape[2] != 3 or image_rgb.size == 0
        ):
            raise ValueError("Scratch detection requires a nonempty uint8 RGB image.")
        height, width = image_rgb.shape[:2]
        gray = np.asarray(Image.fromarray(image_rgb).convert("L"), dtype=np.float32)
        tensor = torch.from_numpy(gray / 127.5 - 1.0)[None, None]

        # Match the library's 256-pixel short side; cap panoramas at 1024 to
        # bound inference memory. Padding keeps the aspect ratio and alignment.
        scale = min(256 / min(height, width), 1024 / max(height, width))
        model_h, model_w = max(1, round(height * scale)), max(1, round(width * scale))
        tensor = F.interpolate(tensor, (model_h, model_w), mode="bilinear", align_corners=False)
        pad_h = max(32, ((model_h + 15) // 16) * 16) - model_h
        pad_w = max(32, ((model_w + 15) // 16) * 16) - model_w
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate").to(self.device)
        logits = self.model(tensor)
        if logits.shape != tensor.shape:
            raise RuntimeError("ZeroScratches returned an unexpected output shape.")
        probability = torch.sigmoid(logits)[0, 0, :model_h, :model_w].cpu().numpy()
        if not np.isfinite(probability).all():
            raise RuntimeError("ZeroScratches returned invalid probabilities.")
        # Nearest interpolation matches the library and retains binary edges.
        probability = cv2.resize(probability, (width, height), interpolation=cv2.INTER_NEAREST)
        mask = (probability >= self.threshold).astype(np.uint8) * 255
        return {
            "mask": mask,
            "threshold": self.threshold,
            "probability_min": float(probability.min()),
            "probability_max": float(probability.max()),
            "probability_mean": float(probability.mean()),
        }
