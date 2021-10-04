from mask_imposer.colored_logger import get_configured_logger
from mask_imposer.definitions import Improvements, Output
from mask_imposer.detector.landmark_detector import Detector
from mask_imposer.imposer.mask_imposer import Imposer
from mask_imposer.input_inspector import Inspector

output = Output(args.output_dir, args.output_format)
inspector.inspect(args.input_dir)
detector.detect()
# detector.save(args.output_dir, args.output_format)
imposer.impose()


class MaskImposer:
    def __init__(self) -> None:
        self._logger = get_configured_logger()
        improvements = Improvements(False, False)
        # mask_set = _create_mask_set(args, logger)
        self._inspector = Inspector(self._logger)
        self._detector = Detector(
            images=self._inspector.get_images(),  # dynamic required
            predictor_fp=None,
            face_detection=True,
            show_samples=False,
            logger=self._logger
        )
        self._imposer = Imposer(
            self._detector.get_landmarks(),
            output,
            mask_set=mask_set,
            improvements=improvements,
            logger=self._logger
        )
