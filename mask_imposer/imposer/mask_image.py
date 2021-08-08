from typing import Dict, Tuple

import cv2
from numpy import ndarray

from mask_imposer.definitions import MaskSet
from mask_imposer.detector.image import Image
from mask_imposer.imposer.mask_pointers import Pointer, PointerMap


class MaskImage(Image):
    def __init__(self, mask_set: MaskSet) -> None:
        super().__init__(mask_set.img_path)
        self._point_map = PointerMap(mask_set.coords_path)

    def resized(self, width: int, height: int, show: bool = False) -> Tuple[ndarray, PointerMap]:
        """We measure distance between opposite points left/right and top/bottom
        on mask and target image and scale mask to make this distances equal.
        """
        x_diff = abs(self._point_map.get_left_point().x - self._point_map.get_right_point().x)
        y_diff = abs(self._point_map.get_bottom_point().y - self._point_map.get_top_point().y)

        x_scale, y_scale = float(width) / float(x_diff), float(height) / float(y_diff)

        curr_h, curr_w, _ = self.img.shape

        new_h, new_w = int(curr_h * y_scale), int(curr_w * x_scale)

        new_im = cv2.resize(self.img, (new_w, new_h))

        if show:
            cv2.imshow("Resized", new_im)
            cv2.waitKey(0)

        return new_im, self._point_map.new_scaled_map(x_scale, y_scale)

    def scale_to(
            self, landmarks_dictionary: Dict[int, Tuple[int, int]]
    ) -> Tuple[ndarray, PointerMap]:
        """ Computes distances to which mask image should be resized and resizes it.

        Accordingly to detected landmarks.
        """
        # responding pointers mask / image
        left = Pointer(*landmarks_dictionary[self._point_map.get_left_index()])
        right = Pointer(*landmarks_dictionary[self._point_map.get_right_index()])
        top = Pointer(*landmarks_dictionary[self._point_map.get_top_index()])
        bottom = Pointer(*landmarks_dictionary[self._point_map.get_bottom_index()])

        height = abs(top.y - bottom.y)
        width = abs(left.x - right.x)

        return self.resized(width, height)
