from logging import Logger
from typing import List, Dict, Optional, Any
from termcolor import colored

import cv2
from mask_imposer.imposer.mask_pointers import Pointer


class CoordinatesCollector:
    """Class allow to input mask characteristic coordinates by clicking them on this image."""

    def __init__(self, mask_img_path: str, logger: Logger) -> None:
        self._logger = logger
        self.window = "Manual collector."
        self.mask_img_path = mask_img_path
        self.img = cv2.imread(mask_img_path)
        self.candidates: List[List[int]] = []
        self.mask_coords: Optional[Dict[int, Pointer]] = None

    def _capture_event(
            self, event: cv2.cuda_Event, x: int, y: int,
            flags: List[Any], params: List[Any]  # pylint:disable=W0613
    ) -> None:
        """Captures coordinates from left/ middle mouse click event inside image."""
        if event in {cv2.EVENT_LBUTTONDBLCLK, cv2.EVENT_MBUTTONDBLCLK} \
                and len(self.candidates) <= 4:
            print(colored("Marked ", "yellow")
                  + colored(f"[{x}, {y}]", "green")
                  + colored(" point.", "yellow"))
            cv2.circle(self.img, (x, y), int(self.img.shape[0] / 40), (255, 0, 0), -1)
            self.candidates.append([x, y])

    def _intercept_coords(self) -> None:
        """Collects coordinates in loop until getting 4 coords pairs."""
        while True:
            cv2.imshow(self.window, self.img)
            cv2.waitKey(1)
            if len(self.candidates) == 4:
                cv2.imshow(self.window, self.img)
                cv2.waitKey(1)
                break

    def collect(self) -> Optional[Dict[int, Pointer]]:
        """Creates json with collected from clickable input coordinates."""
        self._logger.info("Collecting mask coordinates interactively.")
        cv2.namedWindow(self.window)
        cv2.setMouseCallback(self.window, self._capture_event)
        self._intercept_coords()
        self._assign_mask_coords()
        confirmed = self._confirm_coords()
        cv2.destroyAllWindows()
        if not confirmed:
            return self.reset()
        self._logger.info(f"Coordinates accepted by user.")
        return self.mask_coords

    def _assign_mask_coords(self) -> None:
        """Assigns points from candidates to coordinates json."""
        self.mask_coords = {
            2: Pointer(*min(self.candidates, key=lambda x: x[0])),
            9: Pointer(*max(self.candidates, key=lambda x: x[1])),
            16: Pointer(*max(self.candidates, key=lambda x: x[0])),
            29: Pointer(*min(self.candidates, key=lambda x: x[1]))
        }

    @staticmethod
    def _confirm_coords() -> bool:
        """Ask user to confirm inputted coords with empty string, y or yes."""
        response = input(
            colored("Are points correctly assigned?: [Y/n]\n", "yellow")
        )
        return response.lower() in {"y", "yes", ""}

    def reset(self) -> Optional[Dict[int, Pointer]]:
        """Resets collector variables and collects coords once again."""
        self._logger.info("Mask coords collector has been reset.")
        self.mask_coords = None
        self.img = cv2.imread(self.mask_img_path)
        self.candidates = []
        return self.collect()
