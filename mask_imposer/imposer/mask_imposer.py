import cv2
from logging import Logger
from typing import Dict, Tuple

import numpy as np

from .mask_image import MaskImage
from ..detector.image import Image


class Imposer:
    """Class overlay mask image on front face according to detected landmarks."""

    def __init__(self, landmarks: Dict[str, Dict[int, Tuple[int, int]]], logger: Logger) -> None:
        self._logger = logger
        self._landmarks = landmarks
        self.mask = MaskImage()

    def impose(self):
        for image_fp, landmarks_dict in self._landmarks.items():
            print(image_fp)
            print(landmarks_dict)
            img = Image(image_fp)
            print(f"Face shape is {img.img.shape}")
            scaled_mask_img = self.mask.scale_to(landmarks_dict)
            print(f"scaled_mask_img shape is {scaled_mask_img.shape}")

            img.img[0:78, 5:160] = scaled_mask_img
            cv2.imshow("Face", img.img)
            cv2.waitKey(0)
            exit()
