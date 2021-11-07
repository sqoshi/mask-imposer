import os
from pathlib import Path
from typing import List
from unittest import TestCase

import cv2

import mask_imposer


def sorted_by_fn(directory) -> List[str]:
    return sorted(
        [os.path.join(directory, f) for f in os.listdir(directory)],
        key=lambda x: Path(x).name
    )


class PythonPackageTestCase(TestCase):
    """Requires installed mask-imposer as package and shape predictor model."""

    def setUp(self) -> None:
        self.test_dir = Path(os.path.abspath(__file__)).parent
        self.input_dir = os.path.join(f"{self.test_dir}", "..", "data", "input")
        self.test_img_path = os.path.join(self.input_dir, "sample2.jpeg")
        self.mask_path = os.path.join(
            self.test_dir.parent.parent.parent,
            "mask_imposer",
            "bundled",
            "set_01",
            "mask_image.png"
        )
        self.imp = mask_imposer.MaskImposer()

    def test_should_impose_mask_on_image_from_path(self) -> None:
        # mask_img = cv2.cvtColor(cv2.imread(self.mask_path), cv2.COLOR_BGR2GRAY)
        # masked_img = cv2.cvtColor(
        #   self.imp.impose_mask(self.test_img_path, show=False), cv2.COLOR_BGR2GRAY
        # )
        self.imp.impose_mask(self.test_img_path, show=True)

        # heat_map = cv2.matchTemplate(masked_img, mask_img, cv2.TM_CCOEFF_NORMED)
        # heat_map = cv2.matchTemplate(masked_img, mask_img, cv2.TM_CCOEFF_NORMED)

        # h, w = masked_img.shape
        # print(masked_img.shape)
        # print(heat_map.shape)
        # y, x = np.unravel_index(np.argmax(heat_map), heat_map.shape)
        # cv2.rectangle(masked_img, (x, y), (x + w, y + h), (0, 0, 255), 5)
        # cv2.imshow("Match", masked_img)
        # cv2.waitKey(0)

    def test_should_impose_mask_on_image_from_numpy_array(self) -> None:
        self.imp.impose_mask((cv2.imread(self.test_img_path), "fake_mask"), show=True)

    def test_should_switch_mask(self) -> None:
        self.imp.impose_mask((cv2.imread(self.test_img_path), "fake_mask"), show=True)
        self.imp.switch_mask(2)
        self.imp.impose_mask((cv2.imread(self.test_img_path), "fake_mask"), show=True)

    def test_should_set_black_rect_instead_of_mask(self) -> None:
        self.imp.switch_mask(0)
        self.imp.impose_mask((cv2.imread(self.test_img_path), "fake_mask"), show=True)

    def test_should_imposing_results_be_same_for_path_and_numpy_array_way(self):
        m1 = self.imp.impose_mask((cv2.imread(self.test_img_path), "fake_mask"), show=False)
        m2 = self.imp.impose_mask(self.test_img_path, show=False)
        assert (m1 == m2).all()
