from typing import Union
from .abstracts import BaseVectorizer
from .wrappers.vgg_face import VGGFace
from .wrappers.facenet import Facenet
from .wrappers.facenet512 import Facenet512

class VectorizerFactory:
    __vectorizer_map = {
        "vgg-face": VGGFace,
        "facenet": Facenet,
        "facenet512": Facenet512,
    }
    @staticmethod
    def get_vectorizer_class(vectorizer_name: str):
        vectorizer_name = vectorizer_name.lower()
        if vectorizer_name not in VectorizerFactory.__vectorizer_map:
            raise ValueError("Vectorizer not found")
        return VectorizerFactory.__vectorizer_map[vectorizer_name]
    @staticmethod
    def get_vectorizer(vectorizer_name: str) -> 'BaseVectorizer':
        return VectorizerFactory.get_vectorizer_class(vectorizer_name)()
    @staticmethod
    def check_vectorizer(vectorizer: Union[str, 'BaseVectorizer']) -> 'BaseVectorizer':
        if not isinstance(vectorizer, (str, BaseVectorizer)):
            raise ValueError("Vectorizer must be the name of the vectorizer or face vectorizer instance")
        if isinstance(vectorizer, str):
            vectorizer = VectorizerFactory.get_vectorizer(vectorizer)
        return vectorizer
