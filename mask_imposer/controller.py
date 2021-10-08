import os
from pathlib import Path
from typing import List, Union

import cv2
import numpy
from cv2 import waitKey

from mask_imposer.colored_logger import get_configured_logger
from mask_imposer.definitions import Improvements, MaskSet
from mask_imposer.detector.landmark_detector import Detector
from mask_imposer.imposer.mask_imposer import Imposer
from mask_imposer.input_inspector import Inspector


def _get_bundled_mask_set(set_index: int) -> MaskSet:
    """Creates MaskSet object from bundled sets."""
    curr_fp = Path(os.path.dirname(os.path.realpath(__file__))).parent
    return MaskSet(
        os.path.join(curr_fp, f"bundled/set_0{set_index}/mask_image.png"),
        os.path.join(curr_fp, f"bundled/set_0{set_index}/mask_coords.json")
    )


class MaskImposer:
    """Class allow to use project as installable package and impose masks programmatically."""

    def __init__(self, bundled_mask_set_idx: int = 1) -> None:
        self._logger = get_configured_logger()
        mask_set = _get_bundled_mask_set(bundled_mask_set_idx)  # possibility to mix
        self._inspector = Inspector(self._logger)
        self._detector = Detector(
            predictor_fp=None,
            face_detection=True,
            show_samples=False,
            auto_download=True,
            logger=self._logger
        )
        self._imposer = Imposer(
            output=None,
            mask_set=mask_set,
            improvements=Improvements(False, False),
            logger=self._logger
        )

    def impose_mask(self, image: Union[str, List[str]], show=False) -> List[numpy.ndarray]:
        """Imposes mask on image.

        :param image: List of paths to images or single image path
        :param show: if True than displays results of imposing
        :return: list of ndarrays images with imposed masks
        """
        images = [image] if not isinstance(image, list) else image
        self._detector.detect(images)
        masked_images = self._imposer.impose(self._detector.get_landmarks())
        self._detector.forget_landmarks()

        if show:
            for mi in masked_images:
                cv2.imshow("Sample", mi)
                waitKey(5)

        return masked_images
