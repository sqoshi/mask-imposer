from typing import Union, Dict, Tuple, List, Optional

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

    def scaled(self, x_scale: Union[int, float], y_scale: Union[int, float]) -> Tuple[int, int]:
        """Returns scaled coordinates, but not affect current object state."""
        return int(self.x * x_scale), int(self.y * y_scale)

    def scale(self, x_scale: Union[int, float], y_scale: Union[int, float]) -> None:
        """Scale current coordinates."""
        self.x, self.y = self.scaled(x_scale, y_scale)

    def __str__(self) -> str:
        return "Pointer({}, {})".format(colored(self.x, "red"), colored(self.y, "red"))


class PointerMap:
    """Mapping pointers to characteristic mask points."""

    def __init__(self, points: Optional[Dict[int, Pointer]] = None) -> None:
        """Pointers are hardcoded to a default image.

        todo: `input-able` points
        """
        left, right, top, bottom = 2, 16, 29, 9
        self._included_indexes: List[int] = [left, bottom, right, top]  # [2, 9, 16, 29]  # L B R T
        if not points:
            self._points: Dict[int, Pointer] = {
                left: Pointer(20, 90),
                bottom: Pointer(250, 465),
                right: Pointer(475, 90),
                top: Pointer(250, 10)  # or 30
            }
        else:
            self._points: Dict[int, Pointer] = points

    def get_included_indexes(self):
        return self._included_indexes

    def get_left_index(self) -> int:
        """Index of left point."""
        return self._included_indexes[0]

    def get_right_index(self) -> int:
        """Index of right point."""
        return self._included_indexes[2]

    def get_bottom_index(self) -> int:
        """Index of bottom point."""
        return self._included_indexes[1]

    def get_top_index(self) -> int:
        """Index of top point."""
        return self._included_indexes[3]

    def get_left_point(self) -> Pointer:
        """Left point is a one of point from left jaw.

        More details available in readme.
        """
        return self._points[2]

    def get_right_point(self) -> Pointer:
        """Right point is a one of point from right jaw.

        More details available in readme.
        """
        return self._points[16]

    def get_top_point(self) -> Pointer:
        """Top point is any point from middle of nose.

        More details available in readme.
        """
        return self._points[29]

    def get_bottom_point(self) -> Pointer:
        """Bottom point is the lowest point chin.

        More details available in readme.
        """
        return self._points[9]

    def get_top_offset(self) -> int:
        return self.get_top_point().y

    def get_left_offset(self) -> int:
        return self.get_left_point().x

    def updated_points(self, x_scale, y_scale) -> Dict[int, Pointer]:
        """Scales all points by appropriate scales.

        :returns
            a new map of points (does not affect object values)
        """
        new_points = {}
        for k, v in self._points.items():
            new_points[k] = Pointer(*self._points[k].scaled(x_scale, y_scale))
        return new_points

    def new_scaled_map(self, x_scale, y_scale):
        return PointerMap(self.updated_points(x_scale, y_scale))
