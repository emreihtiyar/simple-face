from typing import Union
import cv2
import numpy as np
from PIL import Image

from simple_face.commons.configs import DownloadConfig, PathConfig


def check_image(img: Union[np.ndarray, Image.Image, str])->np.ndarray:
    if isinstance(img, str):
        img = cv2.imread(img)
    if isinstance(img, Image.Image):
        img = np.array(img)
    if not isinstance(img, np.ndarray):
        raise ValueError("Image must be numpy array, PIL image or path to image")
    return img