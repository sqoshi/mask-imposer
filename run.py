import os
from argparse import ArgumentParser, Namespace
from logging import Logger

from mask_imposer.colored_logger import get_configured_logger
from mask_imposer.definitions import ImageFormat, Output, Improvements, MaskSet
from mask_imposer.collector import CoordinatesCollector
from mask_imposer.detector.landmark_detector import Detector
from mask_imposer.imposer.mask_imposer import Imposer
from mask_imposer.input_inspector import Inspector


def get_bundled_mask_set(set_index: int) -> MaskSet:
    """Creates MaskSet object from bundled sets."""
    curr_fp = os.path.dirname(os.path.realpath(__file__))
    return MaskSet(
        os.path.join(curr_fp, f"mask_imposer/bundled/set_0{set_index}/mask_image.png"),
        os.path.join(curr_fp, f"mask_imposer/bundled/set_0{set_index}/mask_coords.json")
    )


def _create_mask_set(args, logger: Logger) -> MaskSet:
    """Determine importance of passed arguments in mask creation procedure."""
    if args.mask_img:
        if args.mask_coords:
            return MaskSet(args.mask_img, args.mask_coords)
        return MaskSet(args.mask_img, CoordinatesCollector(args.mask_img, logger).collect())
    return get_bundled_mask_set(args.use_bundled_mask)


def _parse_args() -> Namespace:
    """Creates parser and parse command line arguments."""
    parser = ArgumentParser()
    parser.add_argument("input_dir", type=str, help="Input directory.")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory.")
    parser.add_argument("--output-format", choices=list(ImageFormat), type=ImageFormat,
                        default=ImageFormat.png,
                        help="Output images format.")
    parser.add_argument("--shape-predictor", type=str, default=None,
                        help="Path to shape predictor.")
    parser.add_argument("--show-samples", action="store_true",
                        help="Show sample after detection.")
    parser.add_argument("--draw-landmarks", default=False, action="store_true",
                        help="Draw circles on detected landmarks coords.")
    parser.add_argument("--off-face-detection", action="store_true",
                        help="Before landmark prediction detect face box.")
    parser.add_argument("--mask-coords", type=str, default=None,
                        # "mask_imposer/bundled/set_01/mask_coords.json",
                        help="Custom mask image path.")
    parser.add_argument("--mask-img", type=str, default=None,
                        help="Custom mask characteristic [2,9,16,29]"
                             " landmarks coordinates json filepath.")
    parser.add_argument("--use-bundled-mask", type=int, default=1, choices=[1, 2],
                        help="Custom mask characteristic [2,9,16,29] "
                             "landmarks coordinates json filepath.")
    return parser.parse_args()


def main():
    logger = get_configured_logger()
    # logger.propagate = False
    # logger.disabled = True
    args = _parse_args()
    improvements = Improvements(args.show_samples, args.draw_landmarks)
    mask_set = _create_mask_set(args, logger)
    output = Output(args.output_dir, args.output_format)
    inspector = Inspector(logger)
    inspector.inspect(args.input_dir)
    detector = Detector(args.shape_predictor, not args.off_face_detection,
                        False, logger)
    detector.detect(inspector.get_images())
    # detector.save(args.output_dir, args.output_format)
    imposer = Imposer(output, mask_set, improvements, logger)
    imposer.impose(detector.get_landmarks())
