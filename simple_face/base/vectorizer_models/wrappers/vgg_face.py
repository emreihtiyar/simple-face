import os
import numpy as np
from simple_face.commons import PathConfig
from simple_face.base.vectorizer_models.architectures import vgg_face
from simple_face.base.vectorizer_models.abstracts import BaseVectorizer
from simple_face.types import VectorizeResult

class VGGFace(BaseVectorizer):
    TARGET_SIZE = (224, 224)
    WEIGHTS_NAME = 'vgg_face_weights.h5'
    # pylint: disable=line-too-long
    WEIGHTS_URL = 'https://github.com/serengil/deepface_models/releases/download/v1.0/vgg_face_weights.h5'
    WEIGHTS_PATH = os.path.join(PathConfig.WEIGHTS_DIR, WEIGHTS_NAME)
    RECOMENDED_THRESHOLDS = {"cosine": 0.40, "euclidean": 0.60, "euclidean_l2": 0.86}
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.build()
    def build(self):
        self._check_weight()
        self._model = vgg_face.base_model()
        self._model.load_weights(self.WEIGHTS_PATH)
    def vectorize(self, img, normalize:bool=True, preprocess:bool=True)->np.ndarray:
        if preprocess:
            img = self.preprocess(img)
        if normalize:
            img = self.normalize(img)
        if "keras" in str(type(self._model)):
            # new tf versions show progress bar and it is annoying
            embedding = self._model.predict(img, verbose=0)[0]
        else:
            # SFace and Dlib are not keras models and no verbose arguments
            embedding = self._model.predict(img)[0]
        return np.array(embedding)