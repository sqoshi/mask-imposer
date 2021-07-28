from logging import DEBUG, Formatter, Logger, StreamHandler
from typing import Any

from termcolor import colored


class ColoredLogger(Logger):
    """Pretties standard logger output depending on type of log."""

    def __init__(self, name: str):
        super().__init__(name)

    def error(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        super().error(colored(msg, "red"), *args, **kwargs)

    def critical(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        super().critical(colored(msg, "red"), *args, **kwargs)

    def warning(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        super().warning(colored(msg, "yellow"), *args, **kwargs)

    def info(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        super().info(colored(msg, "green"), *args, **kwargs)


def get_configured_logger() -> ColoredLogger:
    """Creates and configures logger.

        Level:
            DEBUG
        Format:
            [HH:MM:SS][LOG_TYPE] - message
    """
    logger = ColoredLogger("MaskImposerLogger")
    logger.setLevel(DEBUG)
    ch = StreamHandler()
    ch.setLevel(DEBUG)
    formatter = Formatter(
        f'[{colored("%(asctime)s", "magenta")}][%(levelname)s] - %(message)s',
        "%H:%M:%S"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger
