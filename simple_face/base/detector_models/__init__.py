from typing import Union
from .abstracts import BaseDetector
from .wrappers.mediapipe import MediapipeDetector


class DetectorFactory:
    __detector_map = {
        "mediapipe": MediapipeDetector,
    }
    @staticmethod
    def get_detector_class(detector_name: str):
        detector_name = detector_name.lower()
        if detector_name not in DetectorFactory.__detector_map:
            raise ValueError("Detector not found")
        return DetectorFactory.__detector_map[detector_name]
    @staticmethod
    def get_detector(detector_name: str) -> BaseDetector:
        return DetectorFactory.get_detector_class(detector_name)()
    @staticmethod
    def check_detector(detector: Union[str, BaseDetector]) -> BaseDetector:
        if not isinstance(detector, (str, BaseDetector)):
            raise ValueError("Detector must be string or simple-face detector instance")
        if isinstance(detector, str):
            detector = DetectorFactory.get_detector(detector)
        return detector
