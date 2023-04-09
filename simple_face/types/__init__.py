import abc
import enum
import numpy as np
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass


class BaseType(metaclass=abc.ABCMeta):
    """Base class for all types."""
    pass


@dataclass
class DetectResult(BaseType):
    confidence: float
    detector: Union[str, 'BaseDetector']
    image: np.ndarray
    region: Tuple[int, int, int, int] # x, y, w, h
    left_eye: Optional[Tuple[int, int]] = None
    right_eye: Optional[Tuple[int, int]] = None
    nose: Optional[Tuple[int, int]] = None
    mouth: Optional[Tuple[int, int]] = None
    left_ear: Optional[Tuple[int, int]] = None
    right_ear: Optional[Tuple[int, int]] = None
    def resize(self, size: Tuple[int, int]) -> 'DetectResult':
        raise NotImplementedError
    def rotate(self, angle: int) -> 'DetectResult':
        raise NotImplementedError
    def align(self, size: Tuple[int, int]) -> 'DetectResult':
        raise NotImplementedError

@dataclass
class VectorizeResult(BaseType):
    face: DetectResult
    vector: np.ndarray
    vectorizer: 'BaseVectorizer'

@dataclass
class CompareResult(BaseType):
    class DistanceMetric(enum.Enum):
        COSINE = "cosine"
        EUCLIDEAN = "euclidean"
        EUCLIDEAN_L2 = "euclidean_l2"
    face1: VectorizeResult
    face2: VectorizeResult
    @property
    def cosine_similarity(self) -> float:
        dot_product = np.dot(self.face1.vector, self.face2.vector)
        norm_a = np.linalg.norm(self.face1.vector)
        norm_b = np.linalg.norm(self.face2.vector)
        return 1 - (dot_product / (norm_a * norm_b))
    @property
    def euclidean_distance(self) -> float:
        return self.__calc_euclidean_distance(self.face1.vector, self.face2.vector)
    @property
    def euclidean_l2_distance(self) -> float:
        return self.__calc_euclidean_distance(
            self.__l2_normalize(self.face1.vector),
            self.__l2_normalize(self.face2.vector)
        )
    @property
    def verify(self) -> bool:
        model = self.face1.vectorizer or self.face2.vectorizer
        return self.is_match(
            threshold=model.RECOMENDED_THRESHOLDS.get("cosine"),
            distance_metric=self.DistanceMetric.COSINE
        )
    def is_match(
        self,threshold,
        distance_metric:Union[str, DistanceMetric] = DistanceMetric.COSINE
    ) -> bool:
        if isinstance(distance_metric, str):
            distance_metric = FaceResult.DistanceMetric(distance_metric)
        if distance_metric == self.DistanceMetric.COSINE:
            return self.cosine_similarity < threshold
        elif distance_metric == self.DistanceMetric.EUCLIDEAN:
            return self.euclidean_distance < threshold
        elif distance_metric == self.DistanceMetric.EUCLIDEAN_L2:
            return self.euclidean_l2_distance < threshold
    @staticmethod
    def __calc_euclidean_distance(source:np.ndarray, target:np.ndarray) -> float:
        euclidean_distance =  source - target
        euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
        return np.sqrt(euclidean_distance)
    @staticmethod
    def __l2_normalize(x):
        return x / np.sqrt(np.sum(np.multiply(x, x)))
