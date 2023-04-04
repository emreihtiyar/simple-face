import abc
import cv2
from typing import List, Callable
import numpy as np
from simple_face.types import DetectResult
from simple_face.base.abstracts import BaseModel


class BaseDetector(BaseModel):
    def __init__(self,minimum_confidence:float = 0.7, preprocess:Callable[[np.ndarray], np.ndarray]=None, **kwargs):
        super().__init__(**kwargs)
        self.min_conf = minimum_confidence
        self.__preprocess_function = preprocess or self.__preprocess
    def __preprocess(self, img:np.ndarray)->np.ndarray:
        if len(img.shape) == 4:
                img = img[0]  # e.g. (1, 224, 224, 3) to (224, 224, 3)
        return img
    def preprocess(self, img:np.ndarray)->np.ndarray:
        return self.__preprocess_function(img)
    @abc.abstractmethod
    def build(self):
        raise NotImplementedError
    @abc.abstractmethod
    def detect(self, img:np.ndarray, preprocess:bool=False) -> List[DetectResult]:
        raise NotImplementedError
