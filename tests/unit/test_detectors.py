import os
import sys
import pytest
import cv2
import numpy as np
from PIL import Image

sys.path.append(os.getcwd())
import simple_face

imgs_folder = "tests/data"

detector_name_list = ["mediapipe", "ssd", "retinaface"]
one_face_img_list = [os.path.join(imgs_folder,x) for x in ["single1.jpg", "single2.jpg", "single3.jpg", "single5.jpg", "single6.jpg", "single7.jpg", "single8.jpg", "single9.jpg", "single10.jpg"]]
two_face_img_list = [os.path.join(imgs_folder,x) for x in ["couple.jpg", "couple2.jpg"]]
#Test from detector name
class TestDetectorName:
#Test from detector name
    @pytest.mark.parametrize("detector", detector_name_list)
    def test_stand_detectors(self, detector):
        img = np.zeros((1024, 720, 3), dtype=np.uint8)
        assert simple_face.detect_faces(img, detector=detector) == [], "No face should be detected"

    @pytest.mark.parametrize("detector", ["invalid",1, None, 1.0, [1,2,3]])
    def test_invalid_detector(self, detector):
        img = np.zeros((1024, 720, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            simple_face.detect_faces(img, detector=detector)

    def test_no_detector(self):
        img = np.zeros((1024, 720, 3), dtype=np.uint8)
        with pytest.raises(TypeError):
            simple_face.detect_faces(img)

    @pytest.mark.parametrize("detector", detector_name_list)
    def test_no_face(self, detector):
        img = cv2.imread("tests/data/empty.jpg")
        assert simple_face.detect_faces(img, detector=detector) == [], "No face should be detected"

    @pytest.mark.parametrize("detector", detector_name_list)
    @pytest.mark.parametrize("img_path", one_face_img_list)
    def test_one_face(self, detector, img_path):
        img = cv2.imread(img_path)
        assert len(simple_face.detect_faces(img, detector=detector)) == 1, "One face should be detected"

    @pytest.mark.parametrize("detector", detector_name_list)
    @pytest.mark.parametrize("img_path", one_face_img_list)
    def test_one_face_from_path(self, detector, img_path):
        assert len(simple_face.detect_faces(img_path, detector=detector)) == 1, "One face should be detected"

    @pytest.mark.parametrize("detector", detector_name_list)
    @pytest.mark.parametrize("img_path", one_face_img_list)
    def test_one_face_from_pillow(self, detector, img_path):
        img = Image.open(img_path)
        assert len(simple_face.detect_faces(img, detector=detector)) == 1, "One face should be detected"

    @pytest.mark.parametrize("detector", detector_name_list)
    @pytest.mark.parametrize("img_path", two_face_img_list)
    def test_two_faces(self, detector, img_path):
        img = cv2.imread(img_path)
        assert len(simple_face.detect_faces(img, detector=detector)) == 2, "Two faces should be detected"

#Test from detector object
class TestDetectorObject:
    detector_list = [simple_face.DetectorFactory.get_detector(name) for name in detector_name_list]

    @pytest.mark.parametrize("detector", detector_list)
    def test_stand(self, detector):
        img = np.zeros((1024, 720, 3), dtype=np.uint8)
        assert simple_face.detect_faces(img, detector=detector) == [], "No face should be detected"

    @pytest.mark.parametrize("detector", ["invalid",1, None, 1.0, [1,2,3]])
    def test_invalid_detector(self, detector):
        img = np.zeros((1024, 720, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            simple_face.detect_faces(img, detector=detector)

    @pytest.mark.parametrize("detector", detector_list)
    def test_no_face(self, detector):
        img = cv2.imread("tests/data/empty.jpg")
        assert simple_face.detect_faces(img, detector=detector) == [], "No face should be detected"

    @pytest.mark.parametrize("detector", detector_list)
    @pytest.mark.parametrize("img_path", one_face_img_list)
    def test_one_face(self, detector, img_path):
        img = cv2.imread(img_path)
        assert len(simple_face.detect_faces(img, detector=detector)) == 1, "One face should be detected"

    @pytest.mark.parametrize("detector", detector_list)
    @pytest.mark.parametrize("img_path", one_face_img_list)
    def test_one_face_from_path(self, detector, img_path):
        assert len(simple_face.detect_faces(img_path, detector=detector)) == 1, "One face should be detected"

    @pytest.mark.parametrize("detector", detector_list)
    @pytest.mark.parametrize("img_path", one_face_img_list)
    def test_one_face_from_pillow(self, detector, img_path):
        img = Image.open(img_path)
        assert len(simple_face.detect_faces(img, detector=detector)) == 1, "One face should be detected"

    @pytest.mark.parametrize("detector", detector_list)
    def test_two_faces(self, detector):
        img = cv2.imread("tests/data/couple.jpg")
        assert len(simple_face.detect_faces(img, detector=detector)) == 2, "Two faces should be detected"

#Test from detector class
class TestDetectorClass:
#Test from detector object with custom parameters
    detector_class_list = [simple_face.DetectorFactory.get_detector_class(name) for name in detector_name_list]

    @pytest.mark.parametrize("detector_class", detector_class_list)
    def test_stand(self, detector_class):
        img = np.zeros((1024, 720, 3), dtype=np.uint8)
        detector = detector_class()
        assert simple_face.detect_faces(img, detector=detector) == [], "No face should be detected"

    @pytest.mark.parametrize("detector_class", detector_class_list)
    def test_no_face(self, detector_class):
        img = cv2.imread("tests/data/empty.jpg")
        detector = detector_class()
        assert simple_face.detect_faces(img, detector=detector) == [], "No face should be detected"

    @pytest.mark.parametrize("detector_class", detector_class_list)
    @pytest.mark.parametrize("img_path", one_face_img_list)
    def test_one_face_from_path(self, detector_class, img_path):
        detector = detector_class()
        assert len(simple_face.detect_faces(img_path, detector=detector)) == 1, "One face should be detected"

    @pytest.mark.parametrize("detector_class", detector_class_list)
    @pytest.mark.parametrize("img_path", one_face_img_list)
    def test_one_face_from_pillow(self, detector_class, img_path):
        detector = detector_class()
        img = Image.open(img_path)
        assert len(simple_face.detect_faces(img, detector=detector)) == 1, "One face should be detected"

    @pytest.mark.parametrize("detector_class", detector_class_list)
    def test_with_custom_confidence(self, detector_class):
        img = cv2.imread("tests/data/single1.jpg")
        detector = detector_class(minimum_confidence=0.5)
        assert len(simple_face.detect_faces(img, detector=detector)) >= 1, "One face should be detected"
