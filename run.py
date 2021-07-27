from argparse import ArgumentParser, Namespace

from mask_imposer.colored_logger import get_configured_logger
from mask_imposer.definitions import ImageFormat
from mask_imposer.input_inspector import Inspector


def _parse_args() -> Namespace:
    """Creates parser and parse command line arguments."""
    parser = ArgumentParser()
    parser.add_argument("input_dir", type=str, help="Input directory.")
    # parser.add_argument("--input-dir", type=str, default=None, help="Input directory.")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory.")
    parser.add_argument("--target-ext", choices=list(ImageFormat), type=ImageFormat, default="png",
                        help="Output images format.")
    return parser.parse_args()


def main():
    logger = get_configured_logger()
    logger.warning("test")
    # logger.propagate = False
    # logger.disabled = True
    args = _parse_args()
    Inspector(logger).inspect(args.input_dir)
