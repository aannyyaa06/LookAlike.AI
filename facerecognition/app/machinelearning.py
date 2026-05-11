import numpy as np
import cv2
import sklearn
import pickle
import joblib
from django.conf import settings 
import os

# ============================================================
# FIX: scikit-learn compatibility
# ============================================================
from sklearn.tree._classes import BaseDecisionTree
if not hasattr(BaseDecisionTree, 'monotonic_cst') or 'monotonic_cst' not in BaseDecisionTree.__dict__:
    BaseDecisionTree.monotonic_cst = None

STATIC_DIR = settings.STATIC_DIR


# ---------------- LOAD MODELS ---------------- #

# face detection
face_detector_model = cv2.dnn.readNetFromCaffe(
    os.path.join(STATIC_DIR,'models/deploy.prototxt.txt'),
    os.path.join(STATIC_DIR,'models/res10_300x300_ssd_iter_140000.caffemodel')
)

# feature extraction
face_feature_model = cv2.dnn.readNetFromTorch(
    os.path.join(STATIC_DIR,'models/openface.nn4.small2.v1.t7')
)

# face recognition
face_recognition_model = joblib.load(
    os.path.join(STATIC_DIR,'models/celebrity_identity_model.pkl')
)

# emotion recognition
emotion_recognition_model = joblib.load(
    os.path.join(STATIC_DIR,'models/emotion_recognition_model.pkl')
)


# ---------------- PIPELINE ---------------- #

def pipeline_model(path):
    img = cv2.imread(path)
    image = img.copy()
    h, w = img.shape[:2]

    # face detection
    img_blob = cv2.dnn.blobFromImage(
        img, 1, (300,300), (104,177,123), swapRB=False, crop=False
    )
    face_detector_model.setInput(img_blob)
    detections = face_detector_model.forward()

    machinlearning_results = dict(
        face_detect_score=[],
        face_name=[],
        face_name_score=[],
        emotion_name=[],
        emotion_name_score=[],
        count=[]
    )

    count = 1

    if len(detections) > 0:
        for i, confidence in enumerate(detections[0,0,:,2]):

            if confidence > 0.5:
                box = detections[0,0,i,3:7] * np.array([w,h,w,h])
                startx, starty, endx, endy = box.astype(int)

                cv2.rectangle(image, (startx,starty), (endx,endy), (0,255,0))

                # ---------------- FEATURE EXTRACTION ---------------- #
                face_roi = img[starty:endy, startx:endx]

                face_blob = cv2.dnn.blobFromImage(
                    face_roi, 1/255, (96,96), (0,0,0), swapRB=True, crop=True
                )

                face_feature_model.setInput(face_blob)
                vectors = face_feature_model.forward()

                # FIX: ensure correct shape (1,128)
                vectors = vectors.reshape(1, -1)

                # ---------------- DEBUG (REMOVE LATER) ---------------- #
                print("Input shape:", vectors.shape)
                print("Face model expects:", face_recognition_model.n_features_in_)
                print("Emotion model expects:", emotion_recognition_model.n_features_in_)

                # ---------------- FACE IDENTITY PREDICTION ---------------- #
                face_name = face_recognition_model.predict(vectors)[0]
                face_score = face_recognition_model.predict_proba(vectors).max()

                # ---------------- EMOTION PREDICTION ---------------- #
                try:
                    emotion_name = emotion_recognition_model.predict(vectors)[0]
                    emotion_score = emotion_recognition_model.predict_proba(vectors).max()
                except ValueError as e:
                    print("Emotion model error:", e)
                    emotion_name = "Model Error"
                    emotion_score = 0.0

                # ---------------- DISPLAY ---------------- #
                text_emotion = '{} : {:.0f} %'.format(emotion_name, 100*emotion_score)
                text_face = '{} : {:.0f} %'.format(face_name, 100*face_score)

                # Draw face name above the bounding box
                cv2.putText(
                    image,
                    text_face,
                    (startx, starty - 10),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (0,255,0),
                    2
                )

                # Draw emotion below the bounding box
                cv2.putText(
                    image,
                    text_emotion,
                    (startx, endy + 25),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (255,255,255),
                    2
                )

                # ---------------- SAVE OUTPUT ---------------- #
                cv2.imwrite(
                    os.path.join(settings.MEDIA_ROOT,'ml_output/process.jpg'),
                    image
                )

                cv2.imwrite(
                    os.path.join(settings.MEDIA_ROOT,f'ml_output/roi_{count}.jpg'),
                    face_roi
                )

                # ---------------- STORE RESULTS ---------------- #
                machinlearning_results['count'].append(count)
                machinlearning_results['face_detect_score'].append(confidence)
                machinlearning_results['face_name'].append(face_name)
                machinlearning_results['face_name_score'].append(face_score)
                machinlearning_results['emotion_name'].append(emotion_name)
                machinlearning_results['emotion_name_score'].append(emotion_score)

                count += 1

    return machinlearning_results
