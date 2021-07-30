from typing import Dict, Union, Tuple

import cv2

from mask_imposer.detector.image import Image


class Point:
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

    def update(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

    def scale(self, x_scale: Union[int, float], y_scale: Union[int, float]) -> None:
        self.x = int(self.x * x_scale)
        self.y = int(self.y * y_scale)

    def scaled(self, x_scale: Union[int, float], y_scale: Union[int, float]) -> Tuple[int, int]:
        return int(self.x * x_scale), int(self.y * y_scale)


class MaskImage(Image):
    def __init__(self,  # tmp solution
                 filepath: str = "/home/piotr/Documents/bsc-thesis/mask-imposer/mask_imposer/imposer/mask_image.png"):
        super().__init__(filepath)
        self._landmarks_cords: Dict[int, Point] = {
            2: Point(20, 90),
            9: Point(250, 465),
            16: Point(475, 90),
            29: Point(250, 10)  # or 30
        }
        self._included_indexes = list(self._landmarks_cords.keys())  # [2, 9, 16, 29]  # L B R T
        # cv2.imshow("Original", self.img)
        # cv2.waitKey(0)

    def get_left_index(self) -> int:
        return self._included_indexes[0]

    def get_right_index(self) -> int:
        return self._included_indexes[2]

    def get_bottom_index(self) -> int:
        return self._included_indexes[1]

    def get_top_index(self) -> int:
        return self._included_indexes[3]

    def get_left_point(self) -> Point:
        return self._landmarks_cords[2]

    def get_right_point(self) -> Point:
        return self._landmarks_cords[16]

    def get_top_point(self) -> Point:
        return self._landmarks_cords[29]

    def get_bottom_point(self) -> Point:
        return self._landmarks_cords[9]

    def updated_points(self, x_scale, y_scale) -> Dict[int, Point]:
        new_points = {}
        for k, v in self._landmarks_cords.items():
            new_points[k] = Point(*self._landmarks_cords[k].scaled(x_scale, y_scale))
        return new_points

    def resized(self, width: int, height: int, show: bool = False):
        x_diff = abs(self.get_left_point().x - self.get_right_point().x)
        y_diff = abs(self.get_bottom_point().y - self.get_top_point().y)

        x_scale = float(width) / float(x_diff)
        y_scale = float(height) / float(y_diff)

        curr_h, curr_w, _ = self.img.shape

        new_h = int(curr_h * y_scale)
        new_w = int(curr_w * x_scale)

        new_im = cv2.resize(self.img, (new_w, new_h))

        if show:
            cv2.imshow("Resized", new_im)
            cv2.waitKey(0)

        return new_im, self.updated_points(x_scale, y_scale)

    def scale_to(self, landmarks_dictionary: Dict[int, Tuple[int, int]]):
        left = Point(*landmarks_dictionary[self.get_left_index()])
        right = Point(*landmarks_dictionary[self.get_right_index()])
        top = Point(*landmarks_dictionary[self.get_top_index()])
        bottom = Point(*landmarks_dictionary[self.get_bottom_index()])
        height = abs(top.y - bottom.y)
        width = abs(left.x - right.x)
        return self.resized(width, height)
