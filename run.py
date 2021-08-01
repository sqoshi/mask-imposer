from argparse import ArgumentParser, Namespace

from mask_imposer.colored_logger import get_configured_logger
from mask_imposer.definitions import ImageFormat
from mask_imposer.detector.landmark_detector import Detector
from mask_imposer.imposer.mask_imposer import Imposer
from mask_imposer.input_inspector import Inspector


def _parse_args() -> Namespace:
    """Creates parser and parse command line arguments."""
    parser = ArgumentParser()
    parser.add_argument("input_dir", type=str, help="Input directory.")
    # parser.add_argument("--input-dir", type=str, default=None, help="Input directory.")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory.")
    parser.add_argument("--output-format", choices=list(ImageFormat), type=ImageFormat, default="png",
                        help="Output images format.")
    parser.add_argument("--shape-predictor", type=str, default=None, help="Path to shape predictor.")
    parser.add_argument("--show-samples", type=bool, default=False, help="Show sample after detection.")
    parser.add_argument("--detect-face-boxes", type=bool, default=False,
                        help="Before landmark prediction detect face box.")
    return parser.parse_args()


def main():
    logger = get_configured_logger()
    logger.warning("test")
    # logger.propagate = False
    # logger.disabled = True
    args = _parse_args()

    inspector = Inspector(logger)
    inspector.inspect(args.input_dir)

    detector = Detector(inspector.get_images(), args.shape_predictor, args.detect_face_boxes, args.show_samples, logger)
    detector.detect()
    # detector.save(args.output_dir, args.output_format)

    imposer = Imposer(detector.get_landmarks(), logger)
    imposer.impose()
