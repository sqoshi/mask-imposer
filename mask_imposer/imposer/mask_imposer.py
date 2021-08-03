from logging import Logger
from typing import Any, Dict, Tuple, Union

import cv2
from numpy import ndarray, shape

from mask_imposer.imposer.mask_pointers import Pointer, PointerMap

from ..detector.image import Image
from .mask_image import MaskImage


def non_negative(diff: Union[float, int]) -> Union[float, int]:
    return 0 if diff < 0 else diff


detections_dict = Dict[str, Dict[int, Tuple[int, int]]]


class Imposer:
    """Class overlay mask image on front face according to detected landmarks."""

    def __init__(self, landmarks: detections_dict, logger: Logger) -> None:
        self._logger = logger
        self._landmarks = landmarks
        self.mask = MaskImage()

    @staticmethod
    def determine_goal_cords(
        landmarks_dict: Dict[int, Tuple[int, int]], mask_pointers_map: PointerMap
    ) -> Pointer:
        left_landmark = Pointer(*landmarks_dict[mask_pointers_map.get_left_index()])
        top_landmark = Pointer(*landmarks_dict[mask_pointers_map.get_top_index()])
        x = non_negative(left_landmark.x - mask_pointers_map.get_left_offset())
        y = non_negative(top_landmark.y - mask_pointers_map.get_top_offset())
        return Pointer(int(x), int(y))

    @staticmethod
    def get_shape_surpluses(target: shape, overlay: shape) -> Tuple[Any, Any]:
        return target.shape[0] - overlay.shape[0], target.shape[1] - overlay.shape[1]

    @staticmethod
    def strategic_paste(
        target_image: Image,
        mask_img: ndarray,
        h_surplus: int,
        w_surplus: int,
        left_top_point: Pointer,
        h_mask_limit: int,
        w_mask_limit: int,
    ) -> None:

        alpha_s = mask_img[:h_mask_limit, :w_mask_limit, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(0, 3):
            only_mask = alpha_s * mask_img[:h_mask_limit, :w_mask_limit, c]

            if not h_surplus and not w_surplus:
                target_image.img[left_top_point.y:, left_top_point.x:, c] = (
                    only_mask
                    + alpha_l
                    * target_image.img[left_top_point.y:, left_top_point.x:, c]
                )
            elif h_surplus and not w_surplus:
                target_image.img[
                    left_top_point.y: -h_surplus, left_top_point.x:, c
                ] = (
                    only_mask
                    + alpha_l
                    * target_image.img[
                        left_top_point.y: -h_surplus, left_top_point.x:, c
                    ]
                )
            elif not h_surplus and w_surplus:
                target_image.img[
                    left_top_point.y:, left_top_point.x: -w_surplus, c
                ] = (
                    only_mask
                    + alpha_l
                    * target_image.img[
                        left_top_point.y:, left_top_point.x: -w_surplus, c
                    ]
                )
            elif h_surplus and w_surplus:
                target_image.img[
                    left_top_point.y: -h_surplus, left_top_point.x: -w_surplus, c
                ] = (
                    only_mask
                    + alpha_l
                    * target_image.img[
                        left_top_point.y: -h_surplus, left_top_point.x: -w_surplus, c
                    ]
                )

    def paste_mask(
        self, target_image: Image, landmarks_dict: Dict[int, Tuple[int, int]]
    ) -> None:
        scaled_mask_img, pointer_map = self.mask.scale_to(landmarks_dict)
        left_top_point = self.determine_goal_cords(landmarks_dict, pointer_map)

        target_h, target_w, _ = target_image.img.shape
        replace_box = target_image.img[left_top_point.y:, left_top_point.x:]
        mask_limit_h, mask_limit_w, _ = replace_box.shape

        h_surplus, w_surplus = self.get_shape_surpluses(
            replace_box, scaled_mask_img[:mask_limit_h, :mask_limit_w]
        )

        self.strategic_paste(
            target_image,
            scaled_mask_img,
            h_surplus,
            w_surplus,
            left_top_point,
            mask_limit_h,
            mask_limit_w,
        )

        cv2.imshow("Result", target_image.img)
        cv2.waitKey(0)

    def impose(self) -> None:
        for image_fp, landmarks_dict in self._landmarks.items():
            if "masked" not in image_fp:
                img_obj = Image(image_fp)
                self.paste_mask(img_obj, landmarks_dict)
