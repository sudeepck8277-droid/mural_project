import unittest

import numpy as np

from utils.restoration import RGBColorRestorer


class RestorationTests(unittest.TestCase):
    def setUp(self):
        self.restorer = RGBColorRestorer()

    def test_fills_damage_with_local_rgb_and_preserves_unmasked_pixels(self):
        image = np.full((80, 120, 3), [130, 65, 25], dtype=np.uint8)
        image[:, 60:] = [25, 85, 160]
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[25:45, 20:40] = 255
        mask[25:45, 80:100] = 255
        image[mask > 0] = [230, 230, 230]
        before = image.copy()
        mask_before = mask.copy()

        result = self.restorer.restore(image, mask)

        self.assertEqual(result.shape, image.shape)
        self.assertEqual(result.dtype, np.uint8)
        np.testing.assert_array_equal(result[mask == 0], image[mask == 0])
        np.testing.assert_allclose(result[35, 30], [130, 65, 25], atol=2)
        np.testing.assert_allclose(result[35, 90], [25, 85, 160], atol=2)
        np.testing.assert_array_equal(image, before)
        np.testing.assert_array_equal(mask, mask_before)

    def test_masked_rgb_values_do_not_contaminate_fill(self):
        rng = np.random.default_rng(42)
        image = rng.integers(20, 180, (60, 90, 3), dtype=np.uint8)
        mask = np.zeros((60, 90), dtype=np.uint8)
        mask[20:40, 30:50] = 255
        alternate = image.copy()
        alternate[mask > 0] = 255
        np.testing.assert_array_equal(
            self.restorer.restore(image, mask),
            self.restorer.restore(alternate, mask),
        )

    def test_no_damage_returns_unchanged_copy(self):
        image = np.full((15, 23, 3), 100, dtype=np.uint8)
        result = self.restorer.restore(image, np.zeros((15, 23), dtype=np.uint8))
        np.testing.assert_array_equal(result, image)
        self.assertFalse(np.shares_memory(result, image))

    def test_border_damage_and_boolean_mask(self):
        image = np.full((40, 60, 3), [110, 70, 30], dtype=np.uint8)
        mask = np.zeros((40, 60), dtype=bool)
        mask[:10, :10] = True
        image[mask] = 255
        result = self.restorer.restore(image, mask)
        np.testing.assert_allclose(result[5, 5], [110, 70, 30], atol=2)
        np.testing.assert_array_equal(result[~mask], image[~mask])

    def test_all_damage_reports_missing_color_source(self):
        with self.assertRaisesRegex(ValueError, "no undamaged colors"):
            self.restorer.restore(
                np.zeros((20, 30, 3), dtype=np.uint8),
                np.full((20, 30), 255, dtype=np.uint8),
            )

    def test_mismatched_mask_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "dimensions"):
            self.restorer.restore(
                np.zeros((20, 30, 3), dtype=np.uint8),
                np.zeros((10, 15), dtype=np.uint8),
            )


if __name__ == "__main__":
    unittest.main()
