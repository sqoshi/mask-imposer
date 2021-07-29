import sys
from os import listdir
from os.path import isdir, join
from typing import List

from mask_imposer.colored_logger import ColoredLogger
from mask_imposer.definitions import ImageFormat


class Inspector:
    """Inspect input directory.

    - Check if path is a directory.
    - Find images (collect images from directory)
    - Inform about unrecognized image format
    """

    def __init__(self, logger: ColoredLogger):
        self._logger = logger
        self._images: List[str] = []

    def _find_images(self, path: str) -> List[str]:
        """Find images in given path recognized by their extensions [filename.extension]."""
        all_files = listdir(path)
        if not all_files:
            self._logger.critical(f"{path} is empty.")
            sys.exit()
        return [join(path, f) for f in listdir(path) if ImageFormat.is_image(join(path, f))]

    def get_images(self) -> List[str]:
        return self._images

    def inspect(self, directory_path: str) -> None:
        """Inspect directory in path.

        Checks if:
            - it is a directory
            - there are images inside
        """

        if not isdir(directory_path):
            self._logger.error(f"{directory_path} is not a directory.")
            sys.exit()

        self._images = self._find_images(directory_path)

        if not self._images:
            self._logger.critical(f"Images not found in {directory_path}.")
            sys.exit()
        else:
            self._logger.info(f"Found {len(self._images)} images in {directory_path}.")

        self._logger.info("Inspection finished.")
