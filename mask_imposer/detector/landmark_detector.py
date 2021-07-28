import cv2
from logging import Logger
from typing import List, Optional, Dict

import dlib
from _dlib_pybind11 import get_frontal_face_detector, shape_predictor

from mask_imposer.detector.download import download_predictor


def _shape_to_dict(shape):
    result = {}
    for i in range(0, 68):
        result[i] = (shape.part(i).x, shape.part(i).y)
    if not result:
        raise NotImplementedError()  # raise landmarks not found
    return result


class Image:
    def __init__(self, filepath) -> None:
        self.img = cv2.imread(filepath)
        self.__name = filepath

    def __str__(self):
        return self.__name

    def get_gray_img(self):
        return cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def get_rectangle(self):
        height, width, _ = self.img.shape
        return dlib.rectangle(left=0, top=0, right=width, bottom=height)


class Detector:
    def __init__(self, images: List[str], predictor_fp: Optional[str], logger: Logger):
        self.logger = logger
        self.images = images
        self.detector = get_frontal_face_detector()
        if not predictor_fp:
            predictor_fp = download_predictor(logger)
        self.predictor = shape_predictor(predictor_fp)
        self.should_detect_face_bb = True
        self.landmarks_collection: Dict = {}

    def _detect_face_rect(self, image: Image) -> dlib.rectangle:
        """Find the coordinates of rectangle in which face occurs."""

        if self.should_detect_face_bb:
            rects = self.detector(image.img, 1)
            if not rects:
                self.logger.warning(f"Face rectangle not found. Treating whole image as face box.")
                return image.get_rectangle()
            if len(rects) == 1:
                return rects.pop()
            self.logger.warning(f"Detected multiple faces on image `{image}`. Taking first found.")
            return rects.pop()
        else:
            # whole image as face rectangle (there should be only a center face on image)
            return image.get_rectangle()

    def detect(self):
        for img_path in self.images:
            image = Image(img_path)
            try:
                rect = self._detect_face_rect(image)  # detect rectangles with faces
                shape = self.predictor(image.get_gray_img(), rect)  # detect landmarks
                self.landmarks_collection[image] = _shape_to_dict(shape)
            except NotImplementedError:  # must be changed
                self.logger.warning(f"Landmarks not detected on {image}.")
                continue
