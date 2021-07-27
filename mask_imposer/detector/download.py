from bz2 import BZ2File
from http.client import HTTPException
from urllib.error import URLError, HTTPError
from urllib.request import urlretrieve

from termcolor import colored

from mask_imposer.colored_logger import ColoredLogger


def _unpack_bz2(filepath: str) -> str:
    """Unpack downloaded bz2 file and returns path to content."""
    model_name = filepath.replace(".bz2", ".dat")
    with open(model_name, 'wb') as fw:
        fw.write(BZ2File(filepath).read())
    return model_name


def download_predictor(logger: ColoredLogger, predictor_fp="shape_predictor_68_face_landmarks.bz2"):
    logger.warning("Shape predictor not found.")
    response = input(
        colored(f"Would you like to download ", "green") + colored('64 [MB]', 'red') + colored(" model ?\n",
                                                                                               "green")
    )
    if str(response).lower() in {"", "y", "yes"}:
        try:

            urlretrieve("http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2", predictor_fp)
            # todo: download animation
            return _unpack_bz2(predictor_fp)

        except URLError or HTTPError or HTTPException:
            logger.critical(
                "Error occurred during model download. "
                "Please download model manually and input filepath via arguments.")
            exit()
        except Exception:
            logger.critical(
                "Error occurred during model decompression. "
                "Please download model manually and input filepath via arguments.")
            exit()
    else:
        logger.critical("Shape predictor not found. Detection interrupted.")
        exit()
