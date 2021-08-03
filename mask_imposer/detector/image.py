from typing import Optional

from _dlib_pybind11 import rectangle
from cv2 import COLOR_BGR2BGRA, COLOR_BGR2GRAY, cvtColor, imread


class Image:
    """Hold image data."""

    def __init__(self, filepath: str) -> None:
        self.img = imread(filepath, -1)
        if self.img.shape[-1] == 3:
            self.img = self.converted_rgba()
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

    def converted_rgba(self) -> cvtColor:
        return cvtColor(self.img, COLOR_BGR2BGRA)
