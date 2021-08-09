from argparse import ArgumentParser, Namespace

import cv2

from mask_imposer.colored_logger import get_configured_logger
from mask_imposer.definitions import ImageFormat, Output, Improvements, MaskSet
from mask_imposer.detector.landmark_detector import Detector
from mask_imposer.imposer.mask_imposer import Imposer
from mask_imposer.input_inspector import Inspector


def _parse_args() -> Namespace:
    """Creates parser and parse command line arguments."""
    parser = ArgumentParser()
    parser.add_argument("input_dir", type=str, help="Input directory.")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory.")
    parser.add_argument("--output-format", choices=list(ImageFormat), type=ImageFormat, default=ImageFormat.png,
                        help="Output images format.")
    parser.add_argument("--shape-predictor", type=str, default=None, help="Path to shape predictor.")
    parser.add_argument("--show-samples", action="store_true",
                        help="Show sample after detection.")
    parser.add_argument("--draw-landmarks", default=False, action="store_true",
                        help="Draw circles on detected landmarks coords.")
    parser.add_argument("--detect-face-boxes", type=bool, default=False,
                        help="Before landmark prediction detect face box.")
    parser.add_argument("--mask-coords", type=str, default="mask_imposer/bundled/set_01/mask_coords.json",
                        help="Custom mask image path.")
    parser.add_argument("--mask-img", type=str, default="mask_imposer/bundled/set_01/mask_image.png",
                        help="Custom mask characteristic [2,9,16,29] landmarks coordinates json filepath.")
    return parser.parse_args()


def main():
    logger = get_configured_logger()
    logger.warning("test")
    # logger.propagate = False
    # logger.disabled = True
    args = _parse_args()
    improvements = Improvements(args.show_samples, args.draw_landmarks)
    mask_set = MaskSet(args.mask_img, args.mask_coords)
    # img = cv2.imread(args.mask_img)
    # cv2.imshow("example", img)
    # cv2.waitKey(0)
    # exit()
    output = Output(args.output_dir, args.output_format)

    inspector = Inspector(logger)
    inspector.inspect(args.input_dir)

    detector = Detector(inspector.get_images(), args.shape_predictor, args.detect_face_boxes, False, logger)
    detector.detect()
    # detector.save(args.output_dir, args.output_format)

    imposer = Imposer(detector.get_landmarks(), output, mask_set, improvements, logger)
    imposer.impose()
