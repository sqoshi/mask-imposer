from typing import List, Optional

from _dlib_pybind11 import shape_predictor, get_frontal_face_detector

from mask_imposer.colored_logger import ColoredLogger
from mask_imposer.detector.download import download_predictor


class Detector:
    def __init__(self, images: List[str], predictor_fp: Optional[str], logger: ColoredLogger):
        self.images = images
        self.detector = get_frontal_face_detector()
        self.predictor = shape_predictor(predictor_fp if predictor_fp else download_predictor(logger))
