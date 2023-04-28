import simple_face
from deepface import DeepFace
import cv2
import os

IMG_DATA_PATH = 'tests/data/images'
SAVE_DIR = 'tests/data/'
img_names = list(filter(lambda x: x.endswith('.jpg') or x.endswith('.jpeg'), os.listdir(IMG_DATA_PATH)))

img1 = cv2.imread(os.path.join(IMG_DATA_PATH, img_names[3]))
img2 = cv2.imread(os.path.join(IMG_DATA_PATH, img_names[4]))

face1 = simple_face.detect_faces(img1, detector="mediapipe")[0]
face2 = simple_face.detect_faces(img2, detector="mediapipe")[0]
face1.resize((224, 224))
face2.resize((224, 224))
# face1 = simple_face.detect_and_vectorize(img1)[0]
# face2 = simple_face.detect_and_vectorize(img2)[0]

result = simple_face.compare_faces(face1, face2, vectorizer='facenet')
result_df = DeepFace.verify(face1.image, face2.image, model_name='Facenet', detector_backend='skip', distance_metric='cosine')

cv2.imshow('img1', face1.image)
cv2.imshow('img2', face2.image)
cv2.waitKey(0)
cv2.destroyAllWindows()

from ipdb import set_trace; set_trace()