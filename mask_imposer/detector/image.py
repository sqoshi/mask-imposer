from typing import Optional

from _dlib_pybind11 import rectangle
from cv2 import cvtColor, COLOR_BGR2GRAY, imread


class Image:
    """Hold image data."""

    def __init__(self, filepath: str) -> None:
        self.img = imread(filepath)
        self.__name: str = filepath
        self._gray_img: Optional[cvtColor] = None
        self._rect: Optional[rectangle] = None

    def __str__(self) -> str:
        return self.__name

    def get_gray_img(self) -> cvtColor:
        """Creates if not yet created image in gray scale."""
        if self._gray_img is None:
            self._gray_img = cvtColor(self.img, COLOR_BGR2GRAY)
        return self._gray_img

    def get_rectangle(self) -> rectangle:
        """Creates if not yet created dlib rectangle of within whole image."""
        if self._rect is None:
            height, width, _ = self.img.shape
            self._rect = rectangle(left=0, top=0, right=width, bottom=height)
        return self._rect
