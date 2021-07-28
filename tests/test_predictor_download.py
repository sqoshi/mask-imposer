import os
from logging import getLogger
from unittest import TestCase
from unittest.mock import patch

from mask_imposer.detector.download import download_predictor


def get_directory_files(path):
    return [f for f in os.listdir(path) if os.path.isfile(f)]


class PredictorDownloadTestCase(TestCase):

    # noinspection PyTypeChecker
    @patch("mask_imposer.detector.download.input", lambda _: "y")
    def test_should_download_predictor(self):
        download_predictor(getLogger("test"), "SPZ.bz2")
        self.assertTrue("SPZ.dat" in get_directory_files("."))
        os.remove("SPZ.dat")
        os.remove("SPZ.bz2")
