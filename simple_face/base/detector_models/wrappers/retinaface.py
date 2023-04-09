import numpy as np
from typing import List, Callable
from simple_face.types import DetectResult
from simple_face.base.detector_models.abstracts import BaseDetector
from retinaface import RetinaFace

class RetinafaceDetector(BaseDetector):
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
        return RetinaFace.build_model()
    def detect(self, img:np.ndarray, preprocess:bool=False) -> List[DetectResult]:
        resp = []
        if preprocess:
            img = self.preprocess(img)
        faces = RetinaFace.detect_faces(img, threshold=self.minimum_confidence, model=self._model)
        if faces is None or len(faces) == 0 or (isinstance(faces, tuple) and faces[0].size == 0):
            return resp
        for f_details in faces.values():
            area = f_details.get('facial_area')
            resp.append(
                DetectResult(
                    image=img[area[1]:area[3], area[0]:area[2]],
                    detector=self,
                    region=(area[0], area[1], area[2] - area[0], area[3] - area[1]),
                    confidence=f_details.get('score'),
                    left_eye=tuple(f_details.get('landmarks').get('left_eye')),
                    right_eye=tuple(f_details.get('landmarks').get('right_eye')),
                    nose=tuple(f_details.get('landmarks').get('nose')),
                )
            )
        return resp
