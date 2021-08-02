import cv2
from logging import Logger
from typing import Dict, Tuple, Union, Any

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

    @staticmethod
    def get_shape_surpluses(target, overlay) -> Tuple[Any, Any]:
        return target.shape[0] - overlay.shape[0], target.shape[1] - overlay.shape[1]

    @staticmethod
    def strategic_paste(target_image, mask_img, h_surplus, w_surplus, left_top_point, h_mask_limit, w_mask_limit):
        alpha_s = mask_img[:h_mask_limit, :w_mask_limit, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(0, 3):
            only_mask = alpha_s * mask_img[:h_mask_limit, :w_mask_limit, c]

            if not h_surplus and not w_surplus:
                target_image.img[left_top_point.y:, left_top_point.x:, c] = \
                    only_mask + alpha_l * target_image.img[left_top_point.y:, left_top_point.x:, c]
            elif h_surplus and not w_surplus:
                target_image.img[left_top_point.y:-h_surplus, left_top_point.x:, c] = \
                    only_mask + alpha_l * target_image.img[left_top_point.y:-h_surplus, left_top_point.x:, c]
            elif not h_surplus and w_surplus:
                target_image.img[left_top_point.y:, left_top_point.x:-w_surplus, c] = \
                    only_mask + alpha_l * target_image.img[left_top_point.y:, left_top_point.x:-w_surplus, c]
            elif h_surplus and w_surplus:
                target_image.img[left_top_point.y:-h_surplus, left_top_point.x:-w_surplus, c] = \
                    only_mask + alpha_l * target_image.img[left_top_point.y:-h_surplus, left_top_point.x:-w_surplus, c]

    def paste_mask(self, target_image: Image, landmarks_dict):
        scaled_mask_img, pointer_map = self.mask.scale_to(landmarks_dict)
        left_top_point = self.determine_goal_cords(landmarks_dict, pointer_map)

        target_h, target_w, _ = target_image.img.shape
        replace_box = target_image.img[left_top_point.y:, left_top_point.x:]
        mask_limit_h, mask_limit_w, _ = replace_box.shape

        h_surplus, w_surplus = self.get_shape_surpluses(replace_box, scaled_mask_img[:mask_limit_h, :mask_limit_w])

        self.strategic_paste(target_image, scaled_mask_img,
                             h_surplus, w_surplus, left_top_point,
                             mask_limit_h, mask_limit_w)

        cv2.imshow("Result", target_image.img)
        cv2.waitKey(0)

    def impose(self):
        for image_fp, landmarks_dict in self._landmarks.items():
            if "masked" not in image_fp:
                img_obj = Image(image_fp)
                self.paste_mask(img_obj, landmarks_dict)
