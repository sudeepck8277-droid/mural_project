import os
from pathlib import Path

import cv2
import numpy as np


class RGBColorRestorer:
    """Fill segmented damage from neighboring, undamaged RGB pixels."""

    def restore(self, image_rgb, damage_mask):
        if (
            image_rgb.dtype != np.uint8
            or image_rgb.ndim != 3
            or image_rgb.shape[2] != 3
            or image_rgb.size == 0
        ):
            raise ValueError("Restoration requires a nonempty uint8 RGB image.")
        if damage_mask.ndim != 2 or damage_mask.shape != image_rgb.shape[:2]:
            raise ValueError("The damage mask must match the input image dimensions.")

        region = damage_mask > 0
        restored = image_rgb.copy()
        if not np.any(region):
            return restored
        if np.all(region):
            raise ValueError(
                "The entire image is marked as damaged; "
                "no undamaged colors are available for restoration."
            )

        mask = region.astype(np.uint8) * 255
        # Exclude a narrow rim from color sampling: segmentation boundaries
        # can still contain faded paint. Only the original mask is written back.
        sampling_mask = cv2.dilate(
            mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
        if np.all(sampling_mask):
            sampling_mask = mask

        # Navier-Stokes inpainting propagates surrounding colors and edges
        # into each hole at native resolution. OpenCV processes each channel
        # independently here, so the RGB order stays intact.
        # Seed holes with the nearest intact RGB sample. This also prevents
        # dark artifacts where a masked component touches the image border.
        _, nearest = cv2.distanceTransformWithLabels(
            sampling_mask, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
        )
        source = image_rgb.copy()
        donors = source[sampling_mask == 0]
        excluded = sampling_mask > 0
        source[excluded] = donors[nearest[excluded] - 1]
        filled = cv2.inpaint(source, sampling_mask, 5, cv2.INPAINT_NS)
        restored[region] = filled[region]
        return restored


class MuralRestorer:
    """Pretrained Big-LaMa restoration guided by the damage segmentation mask."""

    name = "Pretrained LaMa"

    def __init__(self, model_path=None, device=None, max_input_dimension=None):
        import torch

        self.torch = torch
        self.model_path = Path(model_path) if model_path else (
            Path(__file__).resolve().parents[1] / "models" / "big-lama.pt"
        )
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"LaMa checkpoint not found: {self.model_path}. "
                "Run python scripts/download_lama.py and restart Flask."
            )
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.max_input_dimension = int(
            max_input_dimension if max_input_dimension is not None else
            os.environ.get("MURAL_LAMA_MAX_INPUT_DIMENSION", "1024")
        )
        if self.max_input_dimension < 64:
            raise ValueError("LaMa's maximum input dimension must be at least 64.")
        # Load once at startup; requests never download models or silently
        # switch to color filling when LaMa is unavailable.
        self.model = torch.jit.load(str(self.model_path), map_location=self.device)
        self.model.eval()
        print(f"Restoration model loaded on {self.device}: {self.name}")

    def restore(self, image_rgb, damage_mask):
        if (
            image_rgb.dtype != np.uint8
            or image_rgb.ndim != 3
            or image_rgb.shape[2] != 3
            or image_rgb.size == 0
        ):
            raise ValueError("Restoration requires a nonempty uint8 RGB image.")
        if damage_mask.ndim != 2 or damage_mask.shape != image_rgb.shape[:2]:
            raise ValueError("The damage mask must match the input image dimensions.")

        region = damage_mask > 0
        restored = image_rgb.copy()
        if not np.any(region):
            return restored
        if np.all(region):
            raise ValueError(
                "The entire image is marked as damaged; "
                "no undamaged context is available for LaMa restoration."
            )

        height, width = image_rgb.shape[:2]
        # Hide faded pixels just beyond the segmentation edge from the model,
        # but composite only inside the original mask to preserve intact paint.
        context_mask = cv2.dilate(
            region.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
        if np.all(context_mask):
            context_mask = region.astype(np.uint8)

        model_image = image_rgb
        if max(height, width) > self.max_input_dimension:
            scale = self.max_input_dimension / max(height, width)
            size = (max(1, round(width * scale)), max(1, round(height * scale)))
            model_image = cv2.resize(image_rgb, size, interpolation=cv2.INTER_AREA)
            # Area coverage keeps thin/single-pixel damage from disappearing.
            context_mask = (cv2.resize(
                context_mask.astype(np.float32), size, interpolation=cv2.INTER_AREA
            ) > 0).astype(np.uint8)
            if np.all(context_mask):
                raise ValueError(
                    "No undamaged context remains at LaMa's processing resolution. "
                    "Increase MURAL_LAMA_MAX_INPUT_DIMENSION."
                )

        model_h, model_w = model_image.shape[:2]
        pad_h = max(64, ((model_h + 7) // 8) * 8) - model_h
        pad_w = max(64, ((model_w + 7) // 8) * 8) - model_w
        # The released TorchScript model accepts RGB [0, 1] BCHW tensors,
        # plus a single-channel mask where 1 means missing. Pad, never stretch.
        model_image = np.pad(
            model_image, ((0, pad_h), (0, pad_w), (0, 0)), mode="symmetric"
        ).astype(np.float32) / 255.0
        model_mask = np.pad(
            context_mask, ((0, pad_h), (0, pad_w)), mode="symmetric"
        ).astype(np.float32)
        model_image[model_mask > 0] = 0
        image_tensor = self.torch.from_numpy(
            np.ascontiguousarray(model_image.transpose(2, 0, 1))
        ).unsqueeze(0).to(self.device)
        mask_tensor = self.torch.from_numpy(model_mask[None, None]).to(self.device)

        with self.torch.inference_mode():
            prediction = self.model(image_tensor, mask_tensor)
        if prediction.shape != image_tensor.shape:
            raise RuntimeError("LaMa returned an unexpected output shape.")
        output = prediction[0, :, :model_h, :model_w].permute(1, 2, 0).cpu().numpy()
        if not np.isfinite(output).all():
            raise RuntimeError("LaMa returned invalid pixel values.")
        output = np.rint(np.clip(output, 0, 1) * 255).astype(np.uint8)
        if (model_h, model_w) != (height, width):
            output = cv2.resize(output, (width, height), interpolation=cv2.INTER_CUBIC)
        restored[region] = output[region]
        return restored
