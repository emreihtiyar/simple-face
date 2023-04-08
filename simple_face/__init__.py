from typing import List, Union
import numpy as np
from PIL import Image
from simple_face.base.detector_models import DetectorFactory
from simple_face.base.vectorizer_models import VectorizerFactory
from simple_face.base.detector_models.abstracts import BaseDetector
from simple_face.types import DetectResult
from simple_face import commons

def detect_faces(
        img: Union[np.ndarray, Image.Image, str],
        detector: Union[str, BaseDetector],
    )-> List[DetectResult]:
    img = commons.check_image(img)
    detector = DetectorFactory.check_detector(detector)
    return detector.detect(img)
