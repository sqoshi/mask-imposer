import os
import shutil
import time
from logging import Logger
from os.path import join
from typing import Any, Dict, Tuple, Union, Optional, List

import cv2
from numpy.typing import NDArray
from termcolor import colored

from mask_imposer.imposer.mask_pointers import Pointer, PointerMap

from ..definitions import ImageFormat, Improvements, MaskSet, Output, Size
from ..detector.image import Image
from .mask_image import MaskImage


def non_negative(diff: Union[float, int]) -> Union[float, int]:
    """Returns 0 if diff is negative or positive value."""
    return 0 if diff < 0 else diff


def get_name_from(path: str) -> str:
    """Gouge filename without format/extension from path."""
    return path.split("/")[-1].split(".")[0]


detections_dict = Dict[str, Dict[int, Tuple[int, int]]]


class Imposer:
    """Class overlay mask image on front face according to detected landmarks."""

    def __init__(
        self,
        output: Optional[Output],
        mask_set: MaskSet,
        improvements: Improvements,
        logger: Logger,
    ) -> None:
        self._logger = logger
        if output is not None:
            self.live_imposing = False
            self._output_dir: str = output.directory
            self._output_format: ImageFormat = output.format
        else:
            self.live_imposing = True
        self._should_draw_landmarks = improvements.draw_landmarks
        self._show_samples = improvements.show_samples
        self._mask = MaskImage(mask_set)

    def switch_mask_set(self, mask_set: MaskSet) -> None:
        """Switches mask image."""
        self._mask = MaskImage(mask_set)

    @staticmethod
    def _fit_left_top_coords(
        landmarks_dict: Dict[int, Tuple[int, int]], mask_pointers_map: PointerMap
    ) -> Pointer:
        """Determines coordinates of left top point from which mask rectangle will be pasted.

        It adjust mask left and top landmarks to facial landmarks from image.
        """
        left_landmark = Pointer(*landmarks_dict[mask_pointers_map.get_left_index()])
        top_landmark = Pointer(*landmarks_dict[mask_pointers_map.get_top_index()])
        x = non_negative(left_landmark.x - mask_pointers_map.get_left_offset())
        y = non_negative(top_landmark.y - mask_pointers_map.get_top_offset())
        return Pointer(int(x), int(y))

    @staticmethod
    def _compute_size_surpluses(
        target: NDArray, overlay: NDArray  # type:ignore
    ) -> Tuple[Any, Any]:
        """Get differences between width and height limit of to-replace box
        from original image and mask.
        """
        return target.shape[0] - overlay.shape[0], target.shape[1] - overlay.shape[1]

    @staticmethod
    def _cut_paste(
        target_image: NDArray[Any],
        mask_img: NDArray[Any],  # type:ignore
        surplus: Size,
        left_top_point: Pointer,
        mask_size: Size,
    ) -> None:
        """Paste mask_img on target_image in specific place on this image.

        Why cut_paste?:
            When left top point of mask image is computed to be placed at some point (X,Y)
            and height or width of mask_image is larger than X/Y + target_image width/height
            than mask image is being cut from the left or bottom.
        """

        alpha_s = mask_img[: mask_size.h, : mask_size.w, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        # channels handling
        for c in range(0, 3):
            only_mask = alpha_s * mask_img[: mask_size.h, : mask_size.w, c]
            if not surplus.h and not surplus.w:
                target_image[left_top_point.y:, left_top_point.x:, c] = (
                    only_mask
                    + alpha_l * target_image[left_top_point.y:, left_top_point.x:, c]
                )
            elif surplus.h and not surplus.w:
                target_image[left_top_point.y: -surplus.h, left_top_point.x:, c] = (
                    only_mask
                    + alpha_l
                    * target_image[left_top_point.y: -surplus.h, left_top_point.x:, c]
                )
            elif not surplus.h and surplus.w:
                target_image[left_top_point.y:, left_top_point.x: -surplus.w, c] = (
                    only_mask
                    + alpha_l
                    * target_image[left_top_point.y:, left_top_point.x: -surplus.w, c]
                )
            elif surplus.h and surplus.w:
                target_image[
                    left_top_point.y: -surplus.h, left_top_point.x: -surplus.w, c
                ] = (
                    only_mask
                    + alpha_l
                    * target_image[
                        left_top_point.y: -surplus.h, left_top_point.x: -surplus.w, c
                    ]
                )

    def _draw_landmarks(
        self,
        landmarks_dict: Dict[int, Tuple[int, int]],
        image: NDArray[Any],
        left_top_point: Pointer,
    ) -> NDArray[Any]:
        """Draw landmarks on output images according to flag."""
        if self._should_draw_landmarks:
            for v in landmarks_dict.values():
                try:
                    image = cv2.circle(image, v, 2, (255, 0, 0), 2)
                except cv2.error:
                    pass
            image = cv2.circle(
                image, (left_top_point.x, left_top_point.y), 2, (255, 0, 0), 2
            )
        return image

    def _paste_mask(
        self, target_image: NDArray[Any], landmarks_dict: Dict[int, Tuple[int, int]]
    ) -> None:
        """Pastes mask image on target in place according to detected landmarks on original image

        Performs different paste strategies depending on request.
        """
        scaled_mask_img, pointer_map = self._mask.scale_to(landmarks_dict)
        left_top_point = self._fit_left_top_coords(landmarks_dict, pointer_map)

        replaced_box_primitive = target_image[left_top_point.y:, left_top_point.x:]
        mask_limits = Size(*replaced_box_primitive.shape[:-1])

        surplus = Size(
            *self._compute_size_surpluses(
                replaced_box_primitive,
                scaled_mask_img[: mask_limits.h, : mask_limits.w],
            )
        )

        self._cut_paste(
            target_image, scaled_mask_img, surplus, left_top_point, mask_limits
        )
        if self._show_samples:
            cv2.imshow(
                "Result",
                self._draw_landmarks(landmarks_dict, target_image, left_top_point),
            )
            cv2.waitKey(0)

    def _determine_safe_output_dir(self) -> None:
        """Check if output dir exists and ask for permission to empty it or create a new one."""
        response = input(
            colored(
                f"Would you like to empty existing directory {self._output_dir}?\n",
                "red",
            )
        )
        if response.lower() in {"y", "yes", ""}:
            shutil.rmtree(self._output_dir)
        elif response.lower() == "new":
            self._output_dir = input(
                colored("Input new output directory name?\n", "green")
            )
        else:
            self._output_dir = self._output_dir + f"{int(time.time())}"

    def _create_output_dir(self) -> None:
        """Creates output directory.

        If directory exists and user does not want to override it modifies output directory.
        """
        while True:
            try:
                os.makedirs(self._output_dir)
                break
            except FileExistsError:
                self._determine_safe_output_dir()

    def save(self, filename: str, image: Image) -> None:
        """Save image with filename in output_format."""
        cv2.imwrite(
            join(self._output_dir, f"{filename}.{self._output_format}"), image.img
        )

    def impose(
        self, landmarks_collection: detections_dict, fake_map
    ) -> Union[NDArray[Any], List[NDArray[Any]]]:
        """Imposes mask image on images stored as a dictionary keys in landmarks detections."""
        if not self.live_imposing:
            self._create_output_dir()
        masked_images = []
        for image_fp, landmarks_dict in landmarks_collection.items():
            if "masked" not in image_fp:
                img_obj = Image(
                    image_fp if fake_map is None else (fake_map[image_fp], image_fp)
                )
                self._paste_mask(img_obj.img, landmarks_dict)
                masked_images.append(img_obj.img)

                if not self.live_imposing:
                    self.save(get_name_from(image_fp), img_obj)

        return masked_images
