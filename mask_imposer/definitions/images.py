from enum import Enum
from os.path import isfile


class ImageFormat(Enum):
    """Available image formats."""

    png = "png"
    jpg = "jpg"
    jpeg = "jpeg"
    webp = "webp"

    def __str__(self) -> str:
        return str(self.value)

    @staticmethod
    def is_image(f: str) -> bool:
        """Check if file is image by checking extension."""
        return isfile(f) and any(
            {str(f).endswith(str(ext)) for ext in list(ImageFormat)}
        )
