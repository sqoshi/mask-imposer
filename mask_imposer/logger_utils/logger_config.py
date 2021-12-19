import logging
from logging import Logger

import coloredlogs


def get_configured_logger() -> Logger:
    """Creates and configures logger.

    Level:
        DEBUG
    Format:
        [HH:MM:SS][LOG_TYPE] - message
    """
    logging.basicConfig(filename="masked_face_recognizer.log")
    logger = logging.getLogger(__name__)
    coloredlogs.install(level="DEBUG")
    return logger
