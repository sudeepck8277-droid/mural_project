import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from utils.restoration import MuralRestorer


class RecordingModel(torch.nn.Module):
    """Distinct channel values make RGB order and compositing errors visible."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, image, mask):
        self.calls.append((image.clone(), mask.clone(), torch.is_grad_enabled()))
        color = torch.tensor([0.8, 0.4, 0.2], device=image.device).view(1, 3, 1, 1)
        return color.expand_as(image)


class LamaRestorationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.checkpoint = Path(self.temp.name) / "model.pt"
        self.checkpoint.touch()
        self.model = RecordingModel()
        with patch("torch.jit.load", return_value=self.model) as loader:
            self.restorer = MuralRestorer(self.checkpoint, device="cpu")
            loader.assert_called_once()

    def test_rgb_mask_padding_and_exact_composite(self):
        image = np.full((71, 109, 3), [30, 70, 120], dtype=np.uint8)
        mask = np.zeros((71, 109), dtype=np.uint8)
        mask[10:30, 40:60] = 255
        before = image.copy()
        result = self.restorer.restore(image, mask)
        self.assertEqual(result.shape, image.shape)
        self.assertEqual(result.dtype, np.uint8)
        np.testing.assert_array_equal(result[mask == 0], image[mask == 0])
        np.testing.assert_array_equal(result[20, 50], [204, 102, 51])
        np.testing.assert_array_equal(image, before)
        self.assertEqual(np.count_nonzero(mask), 400)
        model_image, model_mask, gradients = self.model.calls[0]
        self.assertEqual(tuple(model_image.shape), (1, 3, 72, 112))
        self.assertEqual(tuple(model_mask.shape), (1, 1, 72, 112))
        self.assertEqual(model_mask[0, 0, 20, 50].item(), 1)
        self.assertEqual(model_mask[0, 0, 0, 0].item(), 0)
        np.testing.assert_allclose(model_image[0, :, 0, 0], np.array([30, 70, 120]) / 255)
        self.assertTrue(torch.all(model_image[0, :, 20, 50] == 0))
        self.assertFalse(gradients)
        self.assertFalse(self.model.training)

    def test_no_damage_skips_inference(self):
        image = np.full((37, 53, 3), 100, dtype=np.uint8)
        result = self.restorer.restore(image, np.zeros((37, 53), dtype=bool))
        np.testing.assert_array_equal(result, image)
        self.assertFalse(np.shares_memory(result, image))
        self.assertEqual(self.model.calls, [])

    def test_tiny_image_border_damage_and_boolean_mask(self):
        image = np.full((5, 9, 3), 100, dtype=np.uint8)
        mask = np.zeros((5, 9), dtype=bool)
        mask[0, 0] = True
        result = self.restorer.restore(image, mask)
        self.assertEqual(tuple(self.model.calls[0][0].shape), (1, 3, 64, 64))
        np.testing.assert_array_equal(result[0, 0], [204, 102, 51])
        np.testing.assert_array_equal(result[~mask], image[~mask])

    def test_large_image_retains_single_pixel_damage_and_original_size(self):
        self.restorer.max_input_dimension = 64
        image = np.full((131, 257, 3), 100, dtype=np.uint8)
        mask = np.zeros((131, 257), dtype=np.uint8)
        mask[63, 129] = 255
        result = self.restorer.restore(image, mask)
        self.assertEqual(result.shape, image.shape)
        self.assertGreater(self.model.calls[0][1].sum().item(), 0)
        np.testing.assert_array_equal(result[63, 129], [204, 102, 51])
        np.testing.assert_array_equal(result[mask == 0], image[mask == 0])

    def test_masked_source_colors_are_not_sent_to_model(self):
        image = np.full((80, 96, 3), 100, dtype=np.uint8)
        mask = np.zeros((80, 96), dtype=np.uint8)
        mask[30:50, 40:60] = 255
        self.restorer.restore(image, mask)
        image[mask > 0] = 255
        self.restorer.restore(image, mask)
        torch.testing.assert_close(self.model.calls[0][0], self.model.calls[1][0])

    def test_invalid_inputs_and_missing_model_fail_clearly(self):
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "undamaged context"):
            self.restorer.restore(image, np.ones((20, 30), dtype=bool))
        with self.assertRaisesRegex(ValueError, "dimensions"):
            self.restorer.restore(image, np.zeros((10, 15), dtype=np.uint8))
        with self.assertRaisesRegex(FileNotFoundError, "download_lama.py"):
            MuralRestorer(Path(self.temp.name) / "missing.pt", device="cpu")

    def test_invalid_model_output_is_not_saved_as_black_pixels(self):
        image = np.zeros((80, 96, 3), dtype=np.uint8)
        mask = np.zeros((80, 96), dtype=np.uint8)
        mask[20:40, 20:40] = 255
        self.restorer.model = lambda image, mask: torch.full_like(image, float("nan"))
        with self.assertRaisesRegex(RuntimeError, "invalid pixel values"):
            self.restorer.restore(image, mask)


if __name__ == "__main__":
    unittest.main()
