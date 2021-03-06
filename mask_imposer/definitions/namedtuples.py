from collections import namedtuple

Size = namedtuple("Size", "h w")
Output = namedtuple("Output", "directory format")
Improvements = namedtuple("Improvements", "show_samples draw_landmarks")
MaskSet = namedtuple("MaskSet", "img_path coords_path")
