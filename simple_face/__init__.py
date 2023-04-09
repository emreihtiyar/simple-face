from .base.funtions import (
    DetectorFactory,
    VectorizerFactory,
    detect_faces,
    vectorize_face,
    detect_and_vectorize,
    compare_vectors,
    compare_faces,
    detect_and_compare,
)


__file__ = 'simple_face/__init__.py'
__version__ = '0.0.1'
__author__ = 'Emre ihtiyar (@emreihtiyar)'
__author_email__ = 'emre.ihti1@gmail.com'
__all__ = [
        'detect_faces', 'vectorize_face', 'detect_and_vectorize',
        'compare_faces', 'compare_vectors', 'detect_and_compare',
        'DetectorFactory', 'VectorizerFactory'
    ]
__description__ = 'Simple Face is a simple face detection and recognition library.'
__doc__ = """"""
