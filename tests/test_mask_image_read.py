from logging import getLogger
from unittest import TestCase

from mask_imposer.imposer.mask_imposer import Imposer


class MaskImageReadTestCase(TestCase):
    @classmethod
    def setUp(cls) -> None:
        cls.imposer = Imposer({}, getLogger("test"))

    # def test_should_read_mask_image(self):
    #     self.imposer._read_mask_image()
