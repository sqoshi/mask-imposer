from typing import List, Optional

from _dlib_pybind11 import get_frontal_face_detector, shape_predictor

from mask_imposer.colored_logger import ColoredLogger
from mask_imposer.detector.download import download_predictor


class Detector:
    def __init__(self, images: List[str], predictor_fp: Optional[str], logger: ColoredLogger):
        self.images = images
        self.detector = get_frontal_face_detector()
        if not predictor_fp:
            predictor_fp = download_predictor(logger)
        self.predictor = shape_predictor(predictor_fp)
