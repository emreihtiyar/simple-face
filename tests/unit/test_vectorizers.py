import os
import sys
import pytest
import cv2
import numpy as np

sys.path.append(os.getcwd())
import simple_face

imgs_folder = "tests/data"
face_models_name_list = ["facenet", "facenet512", "vgg-face"]

detector = simple_face.DetectorFactory.get_detector("ssd")


#Test from face model name
class TestVectorizorsName:
    @pytest.mark.parametrize("vectorizor_name", face_models_name_list)
    def test_stand_vectorizors(self, vectorizor_name):
        img = np.zeros((1024, 720, 3), dtype=np.uint8)
        assert simple_face.VectorizerFactory.get_vectorizer(vectorizor_name) is not None, "Face model should not be None"        

    def test_wrong_vectorizor_name(self):
        with pytest.raises(ValueError):
            model = simple_face.VectorizerFactory.get_vectorizer("wrong_name")
            assert model is None, "Face model should be None but model is wrong_name"

    @pytest.mark.parametrize("vectorizor_name", face_models_name_list)
    def test_vectorize_result_type(self, vectorizor_name):
        img = np.zeros((1024, 720, 3), dtype=np.uint8)
        model =  simple_face.VectorizerFactory.get_vectorizer(vectorizor_name)
        assert type(model.vectorize(img, True, True)) == np.ndarray, "Face model should return a numpy array"
