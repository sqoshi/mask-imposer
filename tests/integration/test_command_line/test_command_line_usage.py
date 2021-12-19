import os
import shutil
import sys
from pathlib import Path
from typing import List
from unittest import TestCase
from unittest.mock import patch

import cv2

import run


def sorted_by_fn(directory) -> List[str]:
    return sorted(
        [os.path.join(directory, f) for f in os.listdir(directory)],
        key=lambda x: Path(x).name,
    )


class CommandLineUsageTestCase(TestCase):
    """Requires installed mask-imposer as package and shape predictor model."""

    def setUp(self) -> None:
        self.test_dir = Path(os.path.abspath(__file__)).parent
        self.expected_dir = os.path.join(f"{self.test_dir}", "..", "data", "expected")

    def test_should_impose_masks_on_all_images_from_directory_by_cmd_call(self) -> None:
        results_dir = os.path.join(f"{self.test_dir}", "actual")

        testargs = [
            "",
            os.path.join(f"{self.test_dir}", "..", "data", "input"),
            "--output-dir",
            results_dir,
        ]

        with patch.object(sys, "argv", testargs):
            run.main()

        actual_paths = sorted_by_fn(results_dir)
        expected_paths = sorted_by_fn(self.expected_dir)

        for ap, ep in zip(actual_paths, expected_paths):
            cv2.imshow("Actual", cv2.imread(ap))
            cv2.imshow("Expected", cv2.imread(ep))
            cv2.waitKey(0)

        # assert all(
        #     compareHist(cv2.imread(h1), cv2.imread(h2), cv2.HISTCMP_CORREL) for h1, h2 in
        #     zip(actual_paths, expected_paths)
        # )

        shutil.rmtree(results_dir)
