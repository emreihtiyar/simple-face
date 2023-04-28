from deepface import DeepFace
import simple_face
import os
import cv2
import numpy as np
from pprint import pprint

IMG_DATA_PATH = 'tests/data/images'
SAVE_DIR = 'tests/data/'
img_names = list(filter(lambda x: x.endswith('.jpg') or x.endswith('.jpeg'), os.listdir(IMG_DATA_PATH)))

for img_all in img_names:
    img_name, ext = os.path.splitext(img_all)
    img_path = os.path.join(IMG_DATA_PATH, img_all)
    face = simple_face.detect_faces(img_path, 'mediapipe')
    if len(face) ==0:
        continue
    face = face[0]
    face.image = cv2.cvtColor(face.image, cv2.COLOR_BGR2RGB)
    cv2.imshow('face', face.image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    obj_df = DeepFace.represent(face.image, model_name='Facenet', detector_backend="skip", align=False)
    obj_sf = simple_face.vectorize_face(face, 'facenet', normalize=False, preprocess=True)
    print('DeepFace:', obj_df[0].get('embedding'))
    print('-'*50)
    print('SimpleFace:', obj_sf.vector.tolist())
    print('-'*50)
    print('DeepFace - SimpleFace:', obj_df[0].get('embedding') == obj_sf.vector.tolist())