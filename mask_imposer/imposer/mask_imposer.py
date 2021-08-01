import cv2
from logging import Logger
from typing import Dict, Tuple, Union

from cv2 import addWeighted
from mask_imposer.imposer.mask_pointers import PointerMap, Pointer

from .mask_image import MaskImage
from ..detector.image import Image


def non_negative(diff: Union[float, int]) -> Union[float, int]:
    return 0 if diff < 0 else diff


def select_smaller(val1, val2):
    return val2 if val1 > val2 else val1


detection_dict = Dict[str, Dict[int, Tuple[int, int]]]


class Imposer:
    """Class overlay mask image on front face according to detected landmarks."""

    def __init__(self, landmarks: detection_dict, logger: Logger) -> None:
        self._logger = logger
        self._landmarks = landmarks
        self.mask = MaskImage()

    @staticmethod
    def determine_goal_cords(landmarks_dict, mask_pointers_map: PointerMap):
        left_landmark = Pointer(*landmarks_dict[mask_pointers_map.get_left_index()])
        top_landmark = Pointer(*landmarks_dict[mask_pointers_map.get_top_index()])

        x = non_negative(left_landmark.x - mask_pointers_map.get_left_offset())
        y = non_negative(top_landmark.y - mask_pointers_map.get_top_offset())

        return Pointer(x, y)

    def paste_mask_on(self, target_image: Image, landmarks_dict):
        scaled_mask_img, pointer_map = self.mask.scale_to(landmarks_dict)
        left_top_point = self.determine_goal_cords(landmarks_dict, pointer_map)
        target_h, target_w, _ = target_image.img.shape
        ms_h, ms_w, _ = scaled_mask_img.shape
        replace_box = target_image.img[left_top_point.y:, left_top_point.x:]
        limit_w, limit_h, _ = replace_box.shape

        print(left_top_point)
        # alpha_s = scaled_mask_img[:, :, 3] / 255.0
        # alpha_l = 1.0 - alpha_s
        # for c in range(0, 3):
        print(scaled_mask_img.shape)
        print(scaled_mask_img[:limit_w, :limit_h].shape)
        cv2.imshow("Cut mask", scaled_mask_img[:limit_w, :limit_h])
        cv2.imshow("Face replacement", target_image.img[left_top_point.y:, left_top_point.x:])
        cv2.waitKey(0)
        target_image.img[left_top_point.y:, left_top_point.x:] = scaled_mask_img[:limit_w, :limit_h]
        cv2.imshow("MaskedFace", target_image.img)
        cv2.waitKey(0)

    def impose(self):
        for image_fp, landmarks_dict in self._landmarks.items():
            if "masked" not in image_fp:
                print(image_fp)
                img_obj = Image(image_fp)
                self.paste_mask_on(img_obj, landmarks_dict)
