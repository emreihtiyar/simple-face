import numpy as np
from typing import List, Callable
from simple_face.types import DetectResult
from simple_face.base.detector_models.abstracts import BaseDetector
from retinaface import RetinaFace

class SkipDetector(BaseDetector):
    def __init__(
            self,
            minimum_confidence:float = 0.75,
            preprocess:Callable[[np.ndarray], np.ndarray]=None,
            **kwargs
        ):
        super().__init__(minimum_confidence=minimum_confidence, preprocess=preprocess, **kwargs)
        self.minimum_confidence = minimum_confidence
        self.preprocess = preprocess or super().preprocess
        self._model = self.build()
    def build(self):
        pass
    def detect(self, img:np.ndarray, preprocess:bool=False) -> List[DetectResult]:
        resp = []
        if preprocess:
            img = self.preprocess(img)
        resp.append(DetectResult(
            image=img,
            confidence=1.0,
            region=[0, 0, img.shape[0], img.shape[1]],
            detector=self,
        ))
        return resp
