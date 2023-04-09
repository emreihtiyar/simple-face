from typing import List, Union
import numpy as np
from PIL import Image
from simple_face.base.detector_models import DetectorFactory, BaseDetector
from simple_face.base.vectorizer_models import VectorizerFactory, BaseVectorizer
from simple_face.types import DetectResult, VectorizeResult, CompareResult
from simple_face import commons

def detect_faces(
        img: Union[np.ndarray, Image.Image, str],
        detector: Union[str, BaseDetector],
        preprocess: bool = True,
    )-> List[DetectResult]:
    img = commons.check_image(img)
    detector = DetectorFactory.check_detector(detector)
    return detector.detect(img, preprocess=preprocess)

def vectorize_face(
    img: DetectResult,
    vectorizer: Union[str, BaseVectorizer],
    preprocess: bool = True,
    normalize: bool = False,
    )->VectorizeResult:
    vectorizer = VectorizerFactory.check_vectorizer(vectorizer)
    vector = vectorizer.vectorize(img.image, preprocess=preprocess, normalize=normalize)
    return VectorizeResult(
        face=img,
        vector=vector,
        vectorizer=vectorizer,
    )

def detect_and_vectorize(
    img: Union[np.ndarray, Image.Image, str],
    detector: Union[str, BaseDetector] = 'mediapipe',
    vectorizer: Union[str, BaseVectorizer] = 'facenet',
    preprocess: bool = True,
    normalize: bool = False,
    )-> List[VectorizeResult]:
    faces = detect_faces(img, detector, preprocess=preprocess)
    return [vectorize_face(face, vectorizer, preprocess=preprocess, normalize=normalize) \
        for face in faces]

def compare_vectors(
    vector1: VectorizeResult,
    vector2: VectorizeResult,
    )-> CompareResult:
    return CompareResult(
        face1=vector1,
        face2=vector2,
    )

def compare_faces(
    face1: DetectResult,
    face2: DetectResult,
    vectorizer: Union[str, BaseVectorizer] = 'facenet',
    preprocess: bool = True,
    normalize: bool = False,
    )-> CompareResult:
    vector1 = vectorize_face(face1, vectorizer, preprocess=preprocess, normalize=normalize)
    vector2 = vectorize_face(face2, vectorizer, preprocess=preprocess, normalize=normalize)
    return CompareResult(
        face1=vector1,
        face2=vector2,
    )

def detect_and_compare(
    img1: Union[np.ndarray, Image.Image, str],
    img2: Union[np.ndarray, Image.Image, str],
    detector: Union[str, BaseDetector] = 'mediapipe',
    vectorizer: Union[str, BaseVectorizer] = 'facenet',
    preprocess: bool = True,
    normalize: bool = False,
    )-> List[CompareResult]:
    response = []
    img1_vectors = detect_and_vectorize(img1, detector, vectorizer,
                    preprocess=preprocess, normalize=normalize)
    img2_vectors = detect_and_vectorize(img2, detector, vectorizer,
                    preprocess=preprocess, normalize=normalize)
    for vector1 in img1_vectors:
        for vector2 in img2_vectors:
            response.append(CompareResult(
                face1=vector1,
                face2=vector2,
            ))
    return response