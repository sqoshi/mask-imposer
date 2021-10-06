import os
from pathlib import Path
from typing import List, Union

import cv2
from cv2 import waitKey

from mask_imposer.colored_logger import get_configured_logger
from mask_imposer.definitions import Improvements, Output, MaskSet
from mask_imposer.detector.landmark_detector import Detector
from mask_imposer.imposer.mask_imposer import Imposer
from mask_imposer.input_inspector import Inspector


# from mask_imposer.mask_utils import get_bundled_mask_set

# output = Output(args.output_dir, args.output_format)
# detector.detect()


# detector.save(args.output_dir, args.output_format)

def _get_bundled_mask_set(set_index: int) -> MaskSet:
    """Creates MaskSet object from bundled sets."""
    curr_fp = Path(os.path.dirname(os.path.realpath(__file__))).parent
    return MaskSet(
        os.path.join(curr_fp, f"bundled/set_0{set_index}/mask_image.png"),
        os.path.join(curr_fp, f"bundled/set_0{set_index}/mask_coords.json")
    )


class MaskImposer:
    def __init__(self) -> None:
        self._logger = get_configured_logger()
        improvements = Improvements(False, False)
        mask_set = _get_bundled_mask_set(1)  # possibility to mix
        self._inspector = Inspector(self._logger)
        self._detector = Detector(
            predictor_fp=None,
            face_detection=True,
            show_samples=False,
            logger=self._logger
        )
        self._imposer = Imposer(
            output=None,
            mask_set=mask_set,
            improvements=improvements,
            logger=self._logger
        )

    def impose_mask(self, image: Union[str, List[str]], show=False):
        images = [image] if not isinstance(image, list) else image
        self._detector.detect(images)
        masked_images = self._imposer.impose(self._detector.get_landmarks())
        self._detector.forget_landmarks()
        if show:
            for mi in masked_images:
                cv2.imshow("Sample", mi)
                waitKey(5)
        return masked_images
