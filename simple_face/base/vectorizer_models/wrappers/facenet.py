import os
import numpy as np
from simple_face.commons import PathConfig
from simple_face.base.vectorizer_models.architectures import inception_resnet
from simple_face.base.vectorizer_models.abstracts import BaseVectorizer

class Facenet(BaseVectorizer):
    TARGET_SIZE = (160, 160)
    WEIGHTS_NAME = "facenet_weights.h5"
    # pylint: disable=line-too-long
    WEIGHTS_URL = 'https://github.com/serengil/deepface_models/releases/download/v1.0/facenet_weights.h5'
    WEIGHTS_PATH = os.path.join(PathConfig.WEIGHTS_DIR, WEIGHTS_NAME)
    RECOMENDED_THRESHOLDS = {"cosine": 0.40, "euclidean": 10, "euclidean_l2": 0.80}
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.build()
    def build(self):
        self._check_weight()
        self._model = inception_resnet.InceptionResNetV2(dimension=128)
        self._model.load_weights(self.WEIGHTS_PATH)
    def vectorize(self, img, normalize:bool=True, preprocess:bool=True)->np.ndarray:
        if preprocess:
            img = self.preprocess(img)
        if normalize:
            img = self.normalize(img)
        if "keras" in str(type(self._model)):
            # new tf versions show progress bar and it is annoying
            embedding = self._model.predict(img, verbose=0)[0].tolist()
        else:
            # SFace and Dlib are not keras models and no verbose arguments
            embedding = self._model.predict(img)[0].tolist()
        return np.array(embedding)