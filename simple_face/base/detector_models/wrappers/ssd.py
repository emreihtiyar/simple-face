import os
import cv2
import numpy as np
from typing import List, Callable
from simple_face.commons.configs import PathConfig as PCfg
from simple_face.commons.downloader import download_single_file
from simple_face.types import DetectResult
from simple_face.base.detector_models.abstracts import BaseDetector

class Ssd300x300Detector(BaseDetector):
    WEIGHTS_NAME = 'res10_300x300_ssd_iter_140000.caffemodel'
    WEIGHTS_URL = 'https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel'
    PROTOTXT_NAME = 'deploy.prototxt'
    PROTOTXT_URL = 'https://github.com/opencv/opencv/raw/3.4.0/samples/dnn/face_detector/deploy.prototxt'
    TARGET_SIZE = (300, 300)
    WEIGHTS_PATH = os.path.join(PCfg.WEIGHTS_DIR, WEIGHTS_NAME)
    PROTOTXT_PATH = os.path.join(PCfg.WEIGHTS_DIR, PROTOTXT_NAME)
    def __init__(self,minimum_confidence:float = 0.99, preprocess:Callable[[np.ndarray], np.ndarray]=None, **kwargs):
        super().__init__(minimum_confidence=minimum_confidence, preprocess=preprocess, **kwargs)
        self.minimum_confidence = minimum_confidence
        self.preprocess = preprocess or self.__preprocess
        self._model = self.build()
    @classmethod
    def _check_weight(cls):
        if cls.WEIGHTS_NAME is None or cls.WEIGHTS_URL is None:
            assert False, "You must define WEIGHTS_NAME and WEIGHTS_URL"
        if not os.path.exists(os.path.join(PCfg.WEIGHTS_DIR, cls.WEIGHTS_NAME)):
            download_single_file(cls.WEIGHTS_NAME, cls.WEIGHTS_URL, PCfg.WEIGHTS_DIR)
        if not os.path.exists(os.path.join(PCfg.WEIGHTS_DIR, cls.PROTOTXT_NAME)):
            download_single_file(cls.PROTOTXT_NAME, cls.PROTOTXT_URL, PCfg.WEIGHTS_DIR)
    def __preprocess(self, img:np.ndarray)->np.ndarray:
        if len(img.shape) == 4:
                img = img[0]  # e.g. (1, 224, 224, 3) to (224, 224, 3)
        if len(img.shape) == 3:
            img = cv2.resize(img, self.TARGET_SIZE)
            img = np.expand_dims(img, axis=0)
        if img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img
    def build(self):
        self._check_weight()
        return cv2.dnn.readNetFromCaffe(
            os.path.join(PCfg.WEIGHTS_DIR, self.PROTOTXT_NAME),
            os.path.join(PCfg.WEIGHTS_DIR, self.WEIGHTS_NAME)
        )
    def detect(self, img:np.ndarray, preprocess:bool=False) -> List[DetectResult]:
        if preprocess:
            img = self.preprocess(img)
        resp = []
        w, h, _ = img.shape
        base_img = img.copy()
        blob = cv2.dnn.blobFromImage(image=cv2.resize(img, (300, 300)))
        self._model.setInput(blob)
        detections = self._model.forward()
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.minimum_confidence:
                box = detections[0, 0, i, 3:7] * np.array([h, w, h, w])
                (x1, y1, x2, y2) = box.astype("int")
                resp.append(
                    DetectResult(
                        image=base_img[y1:y2, x1:x2],
                        detector=self,
                        region=(x1, y1, x2-x1, y2-y1),
                        confidence=confidence
                    )
                )

        return resp