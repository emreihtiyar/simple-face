import abc
from typing import Callable
import cv2
import numpy as np
from simple_face.base.abstracts import BaseModel

import abc
import cv2
from typing import List, Callable
import numpy as np
from simple_face.types import DetectResult
from simple_face.base.abstracts import BaseModel


class BaseVectorizer(BaseModel):
    TARGET_SIZE = None
    RECOMENDED_THRESHOLDS = {"cosine": 0.40, "euclidean": 0.55, "euclidean_l2": 0.75}
    def __init__(self, normalize:Callable[[np.ndarray], np.ndarray]=None, preprocess:Callable[[np.ndarray], np.ndarray]=None, **kwargs):
        super().__init__(**kwargs)
        self.__normalize_function = normalize or self.__normalize
        self.__preprocess_function = preprocess or self.__preprocess
    def __normalize(self, img:np.ndarray)->np.ndarray:
        mean, std = img.mean(), img.std()
        img = (img - mean) / std
        return img
    def __preprocess(self, img:np.ndarray)->np.ndarray:
        if len(img.shape) == 4:
                img = img[0]  # e.g. (1, 224, 224, 3) to (224, 224, 3)
        if len(img.shape) == 3:
            img = cv2.resize(img, self.TARGET_SIZE)
            img = np.expand_dims(img, axis=0)
        return img
    def normalize(self, img:np.ndarray)->np.ndarray:
        return self.__normalize_function(img)
    def preprocess(self, img:np.ndarray)->np.ndarray:
        return self.__preprocess_function(img)
    @abc.abstractmethod
    def build(self):
        raise NotImplementedError
    @abc.abstractmethod
    def vectorize(self, img:np.ndarray, normalize:bool=True, preprocess:bool=True)->np.ndarray:
        raise NotImplementedError
