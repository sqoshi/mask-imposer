import sys
from bz2 import BZ2File
from http.client import HTTPException
from logging import Logger
from tarfile import CompressionError
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

from termcolor import colored

from mask_imposer.beautifiers import TerminalProgressBar


def _unpack_bz2(filepath: str) -> str:
    """Unpack downloaded bz2 file and returns path to content."""
    model_name = filepath.replace(".bz2", ".dat")
    with open(model_name, "wb") as fw:
        fw.write(BZ2File(filepath).read())
    return model_name


def _accepted_download() -> bool:
    """Ask for permission to download bundled model."""
    response = input(
        colored("Would you like to download ", "green")
        + colored("64 [MB]", "red")
        + colored(" model ?\n", "green")
    )
    return str(response).lower() in {"", "y", "yes"}


def download_predictor(
    logger: Logger,
    url: str = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
    predictor_fp: str = "shape_predictor_68_face_landmarks.bz2"
) -> str:
    """Downloads default dlib shape predictor (68-landmark)"""

    logger.warning("Shape predictor not found.")
    if _accepted_download():
        try:
            urlretrieve(url, predictor_fp, TerminalProgressBar())
            return _unpack_bz2(predictor_fp)
        except URLError or HTTPError or HTTPException:
            logger.critical(
                "Error occurred during model download. "
                "Please download model manually and input filepath via arguments."
            )
            sys.exit()
        except FileNotFoundError or FileExistsError or CompressionError:
            logger.critical(
                "Error occurred during model decompression. "
                "Please input filepath to model via terminal arguments."
            )
            sys.exit()
    else:
        logger.critical("Shape predictor not provided. Detection interrupted.")
        sys.exit()
