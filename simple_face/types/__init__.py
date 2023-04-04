import abc
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
    face_image: np.ndarray
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