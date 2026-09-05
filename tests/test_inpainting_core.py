import unittest

import numpy as np
from PIL import Image

from cvaa.inpainting import (
    adaptive_expand_mask,
    count_outside_mask_changed_pixels,
    exact_composite,
)


class TestInpaintingCore(unittest.TestCase):
    def test_exact_composite_preserves_outside_mask(self):
        source = np.zeros((16, 16, 3), dtype=np.uint8)
        generated = np.full((16, 16, 3), 255, dtype=np.uint8)
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[4:8, 4:8] = 255

        result = exact_composite(
            Image.fromarray(source),
            Image.fromarray(generated),
            mask,
        )

        changed = count_outside_mask_changed_pixels(
            Image.fromarray(source),
            result,
            mask,
        )
        self.assertEqual(changed, 0)

    def test_adaptive_mask_contains_exact_mask(self):
        exact = np.zeros((32, 32), dtype=np.uint8)
        exact[12:20, 12:20] = 255

        expanded, radius, _ = adaptive_expand_mask(
            exact,
            ratio=0.35,
            min_px=2,
            max_px=8,
        )

        self.assertGreaterEqual(radius, 2)
        self.assertTrue(
            np.all(expanded[exact > 0] == 255)
        )


if __name__ == "__main__":
    unittest.main()
