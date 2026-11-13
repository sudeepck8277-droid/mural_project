import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np
import torch

from utils.scratch_segmentation import ScratchSegmenter
from utils.segmentation import CombinedDamageSegmenter
from utils.pipeline import MuralRestorationPipeline


def detector_result(mask):
    return {
        "mask": mask, "threshold": 0.4, "probability_min": 0.1,
        "probability_mean": 0.2, "probability_max": 0.9,
    }


class CombinedSegmentationTests(unittest.TestCase):
    def make_segmenter(self, trained_mask, scratch_mask):
        segmenter = CombinedDamageSegmenter.__new__(CombinedDamageSegmenter)
        segmenter.trained = Mock()
        segmenter.scratches = Mock()
        segmenter.trained.predict.return_value = detector_result(trained_mask)
        segmenter.scratches.predict.return_value = detector_result(scratch_mask)
        return segmenter

    def test_union_includes_both_models_and_counts_overlap_once(self):
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        trained = np.zeros((10, 10), dtype=np.uint8)
        scratch = trained.copy()
        trained[1:3, 1:3] = 255
        scratch[2:4, 2:4] = 255
        segmenter = self.make_segmenter(trained, scratch)
        result = segmenter.predict(image)
        self.assertEqual(np.count_nonzero(result['mask']), 7)
        self.assertEqual(result['mask'].dtype, np.uint8)
        self.assertEqual(result['detection']['trained_percentage'], 4)
        self.assertEqual(result['detection']['scratch_percentage'], 4)
        self.assertEqual(result['detection']['additional_percentage'], 3)
        np.testing.assert_array_equal(result['trained_mask'], trained)
        np.testing.assert_array_equal(result['scratch_mask'], scratch)
        self.assertIs(segmenter.trained.predict.call_args.args[0], image)
        self.assertIs(segmenter.scratches.predict.call_args.args[0], image)

    def test_union_handles_empty_boolean_and_overlapping_masks(self):
        empty = np.zeros((8, 9), dtype=bool)
        damage = empty.copy()
        damage[3, 4] = True
        for left, right in [(damage, empty), (empty, damage), (damage, damage)]:
            result = CombinedDamageSegmenter.combine_masks(left, right)
            self.assertEqual(np.count_nonzero(result), 1)
            self.assertEqual(result[3, 4], 255)
        self.assertFalse(CombinedDamageSegmenter.combine_masks(empty, empty).any())

    def test_misaligned_masks_fail_before_restoration(self):
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            CombinedDamageSegmenter.combine_masks(np.zeros((10, 10)), np.zeros((5, 5)))
        segmenter = self.make_segmenter(np.zeros((5, 5)), np.zeros((5, 5)))
        with self.assertRaisesRegex(ValueError, 'align'):
            segmenter.predict(np.zeros((10, 10, 3), dtype=np.uint8))

    def test_scratch_only_damage_reaches_lama_and_saved_masks_match(self):
        image = np.full((40, 60, 3), 60, dtype=np.uint8)
        trained = np.zeros((40, 60), dtype=np.uint8)
        scratch = trained.copy()
        scratch[10:20, 30:40] = 255
        pipeline = MuralRestorationPipeline.__new__(MuralRestorationPipeline)
        pipeline.segmenter = self.make_segmenter(trained, scratch)
        pipeline.restorer = Mock(name='restorer')
        pipeline.restorer.name = 'Pretrained LaMa'
        pipeline.restorer.restore.side_effect = lambda rgb, mask: rgb.copy()
        pipeline.super_resolver = Mock()
        pipeline.super_resolver.upscale.side_effect = lambda rgb: rgb.copy()
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'input.png'
            cv2.imwrite(str(source), image)
            result = pipeline.process(str(source), folder, 'test')
            pipeline.restorer.restore.assert_called_once()
            np.testing.assert_array_equal(pipeline.restorer.restore.call_args.args[1], scratch)
            np.testing.assert_array_equal(cv2.imread(result['mask_path'], 0), scratch)
            np.testing.assert_array_equal(cv2.imread(result['trained_mask_path'], 0), trained)
            np.testing.assert_array_equal(cv2.imread(result['scratch_mask_path'], 0), scratch)
            self.assertEqual(result['damage_percentage'], 4.17)


class RecordingScratchModel(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value
        self.last_input = None

    def forward(self, image):
        self.last_input = image.clone()
        return torch.full_like(image, self.value)


class ScratchDetectorTests(unittest.TestCase):
    def make_detector(self, logit):
        detector = ScratchSegmenter.__new__(ScratchSegmenter)
        detector.threshold = 0.4
        detector.device = torch.device('cpu')
        detector.model = RecordingScratchModel(logit)
        return detector

    def test_all_positive_mask_keeps_white_and_original_odd_dimensions(self):
        detector = self.make_detector(10)
        image = np.full((73, 117, 3), 255, dtype=np.uint8)
        result = detector.predict(image)
        self.assertEqual(result['mask'].shape, image.shape[:2])
        self.assertTrue(np.all(result['mask'] == 255))
        tensor = detector.model.last_input
        self.assertEqual(tensor.shape[1], 1)
        self.assertEqual(tensor.shape[2] % 16, 0)
        self.assertEqual(tensor.shape[3] % 16, 0)
        self.assertTrue(torch.all(tensor == 1))

    def test_negative_mask_is_empty_and_black_normalizes_to_minus_one(self):
        detector = self.make_detector(-10)
        result = detector.predict(np.zeros((32, 64, 3), dtype=np.uint8))
        self.assertFalse(result['mask'].any())
        self.assertTrue(torch.all(detector.model.last_input == -1))

    def test_tiny_panorama_is_bounded_and_restored_to_original_coordinates(self):
        detector = self.make_detector(10)
        result = detector.predict(np.zeros((1, 2049, 3), dtype=np.uint8))
        self.assertEqual(result['mask'].shape, (1, 2049))
        self.assertLessEqual(max(detector.model.last_input.shape), 1024)

    def test_invalid_image_and_probabilities_are_rejected(self):
        detector = self.make_detector(float('nan'))
        with self.assertRaisesRegex(ValueError, 'RGB'):
            detector.predict(np.zeros((10, 10), dtype=np.uint8))
        with self.assertRaisesRegex(RuntimeError, 'invalid probabilities'):
            detector.predict(np.zeros((10, 10, 3), dtype=np.uint8))


if __name__ == '__main__':
    unittest.main()
