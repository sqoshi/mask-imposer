import os
from pathlib import Path
from typing import Any, List, Union

import cv2
from cv2 import waitKey
from landmarks_predictor import LandmarksPredictor
from numpy.typing import NDArray

from mask_imposer.definitions import Improvements, MaskSet
from mask_imposer.imposer.mask_imposer import Imposer
from mask_imposer.input_inspector import Inspector
from mask_imposer.logger_utils import get_configured_logger


def _get_bundled_mask_set(set_index: int) -> MaskSet:
    """Creates MaskSet object from bundled sets."""
    curr_fp = Path(os.path.dirname(os.path.realpath(__file__))).parent
    return MaskSet(
        os.path.join(curr_fp, f"mask_imposer/bundled/set_0{set_index}/mask_image.png"),
        os.path.join(
            curr_fp, f"mask_imposer/bundled/set_0{set_index}/mask_coords.json"
        ),
    )


class MaskImposer:
    """Class allow to use project as installable package and impose masks programmatically."""

    def __init__(self, bundled_mask_set_idx: int = 1) -> None:
        self._logger = get_configured_logger()
        mask_set = _get_bundled_mask_set(bundled_mask_set_idx)  # possibility to mix
        self._inspector = Inspector(self._logger)
        self._detector = LandmarksPredictor(
            predictor_fp=None,
            face_detection=True,
            show_samples=False,
            auto_download=True,
        )
        self._imposer = Imposer(
            output=None,
            mask_set=mask_set,
            improvements=Improvements(False, False),
            logger=self._logger,
        )

    def switch_mask(self, bundled_mask_set_idx: int) -> None:
        """Switches mask set in mask imposer."""
        self._imposer.switch_mask_set(_get_bundled_mask_set(bundled_mask_set_idx))

    def impose_mask(
            self, image: Union[str, List[str]], show: bool = False
    ) -> Union[NDArray[Any], List[NDArray[Any]]]:
        """Imposes mask on image.

        :param image: List of paths to images or single image path
        :param show: if True than displays results of imposing
        :return: list of ndarrays images with imposed masks
        """
        self._logger.info("Mask imposing procedure started.")
        images = [image] if not isinstance(image, list) else image
        self._detector.detect(images, create_map=True)
        masked_images = self._imposer.impose(
            self._detector.get_landmarks(), self._detector.fake_map
        )
        self._detector.forget_landmarks()

        if show:
            for mi in masked_images:
                cv2.imshow("Sample", mi)
                waitKey(0)

        self._logger.info("Mask imposing procedure finished. ")

        # single image was passed then return its result instead of list
        if len(masked_images) == 1:
            return masked_images.pop()

        return masked_images

    @classmethod
    def save(cls, img: NDArray[Any], filepath: str) -> None:
        """Saves image in given path using opencv."""
        cv2.imwrite(filepath, img)
