from typing import Dict, List, Optional, Tuple, Union

from termcolor import colored


class Pointer:
    """Coordinates x,y wrapper*."""

    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

    def update(self, x: int, y: int) -> None:
        """Update current coordinates."""
        self.x = x
        self.y = y

    def scaled(
            self, x_scale: Union[int, float], y_scale: Union[int, float]
    ) -> Tuple[int, int]:
        """Returns scaled coordinates, but not affect current object state."""
        return int(self.x * x_scale), int(self.y * y_scale)

    def scale(self, x_scale: Union[int, float], y_scale: Union[int, float]) -> None:
        """Scale current coordinates."""
        self.x, self.y = self.scaled(x_scale, y_scale)

    def __str__(self) -> str:
        return "Pointer({}, {})".format(
            colored(f"{self.x}", "red"), colored(f"{self.y}", "red")
        )


class PointerMap:
    """Mapping pointers to characteristic mask points."""

    def __init__(self, input_points: Optional[Dict[int, Pointer]] = None) -> None:
        """Pointers are hardcoded to a default image."""
        # [2, 9, 16, 29]  # L B R T

        # hardcoded by standard mask image.
        self._left_index = 2
        self._right_index = 16
        self._top_index = 29
        self._bottom_index = 9
        if not input_points:
            self._points: Dict[int, Pointer] = {
                self._left_index: Pointer(20, 90),
                self._bottom_index: Pointer(250, 465),
                self._right_index: Pointer(475, 90),
                self._top_index: Pointer(250, 10)  # or 30
            }
        else:
            self._points: Dict[int, Pointer] = input_points

    def get_included_indexes(self) -> List[int]:
        """List of used landmarks indexes (according to readme landmarks scheme)."""
        return [
            self._left_index,
            self._right_index,
            self._top_index,
            self._bottom_index,
        ]

    def get_left_index(self) -> int:
        """Index of left point."""
        return self._left_index

    def get_right_index(self) -> int:
        """Index of right point."""
        return self._right_index

    def get_bottom_index(self) -> int:
        """Index of bottom point."""
        return self._bottom_index

    def get_top_index(self) -> int:
        """Index of top point."""
        return self._top_index

    def get_left_point(self) -> Pointer:
        """Left point is a one of point from left jaw.

        More details available in readme.
        """
        return self._points[self.get_left_index()]

    def get_right_point(self) -> Pointer:
        """Right point is a one of point from right jaw.

        More details available in readme.
        """
        return self._points[self.get_right_index()]

    def get_top_point(self) -> Pointer:
        """Top point is any point from middle of nose.

        More details available in readme.
        """
        return self._points[self.get_top_index()]

    def get_bottom_point(self) -> Pointer:
        """Bottom point is the lowest point chin.

        More details available in readme.
        """
        return self._points[self.get_bottom_index()]

    def get_top_offset(self) -> int:
        """Distance from top image limit to highest (top) point."""
        return self.get_top_point().y

    def get_left_offset(self) -> int:
        """Distance from left image limit to the closest to the left point."""
        return self.get_left_point().x

    def updated_points(
            self, x_scale: Union[int, float], y_scale: Union[int, float]
    ) -> Dict[int, Pointer]:
        """Scales all points by appropriate scales.

        :returns
            a new map of points (does not affect object values)
        """
        new_points = {}
        for k, v in self._points.items():
            new_points[k] = Pointer(*self._points[k].scaled(x_scale, y_scale))
        return new_points

    def new_scaled_map(  # type:ignore
            self, x_scale: Union[int, float], y_scale: Union[int, float]
    ):
        """Builds a new map scaled to given scales."""
        return PointerMap(self.updated_points(x_scale, y_scale))
