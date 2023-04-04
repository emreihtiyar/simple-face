import cv2
import numpy as np
import mediapipe as mp
from typing import List, Callable
from simple_face.base.detector_models.abstracts import BaseDetector
from simple_face.types import DetectResult

class MediapipeDetector(BaseDetector):
    def __init__(
            self,
            minimum_confidence:float = 0.7,
            preprocess:Callable[[np.ndarray], np.ndarray]=None,
            **kwargs
        ):
        super().__init__(minimum_confidence, preprocess, **kwargs)
        self._model = self.build()
    def build(self):
        return mp.solutions.face_detection.FaceDetection(self.min_conf)
    def detect(self, img:np.ndarray, preprocess:bool=False) -> List[DetectResult]:
        if preprocess:
            img = self.preprocess(img)
        resp = []
        img_height, img_width, _ = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self._model.process(img)
        if results.detections:
            for detection in results.detections:
                (confidence,) = detection.score
                if confidence < self.min_conf:
                    continue
                bounding_box = detection.location_data.relative_bounding_box
                landmarks = detection.location_data.relative_keypoints
                x = int(bounding_box.xmin * img_width)
                w = int(bounding_box.width * img_width)
                y = int(bounding_box.ymin * img_height)
                h = int(bounding_box.height * img_height)
                right_eye = (int(landmarks[0].x * img_width), int(landmarks[0].y * img_height))
                left_eye = (int(landmarks[1].x * img_width), int(landmarks[1].y * img_height))
                nose = (int(landmarks[2].x * img_width), int(landmarks[2].y * img_height))
                mouth = (int(landmarks[3].x * img_width), int(landmarks[3].y * img_height))
                right_ear = (int(landmarks[4].x * img_width), int(landmarks[4].y * img_height))
                left_ear = (int(landmarks[5].x * img_width), int(landmarks[5].y * img_height))
                if x > 0 and y > 0:
                    resp.append(DetectResult(
                        confidence=confidence,
                        detector=self,
                        face_image=img[y : y + h, x : x + w],
                        region=(x, y, w, h),
                        right_eye=right_eye,
                        left_eye=left_eye,
                        nose=nose,
                        mouth=mouth,
                        right_ear=right_ear,
                        left_ear=left_ear
                    ))
        return resp