"""
=============================================================
  FACE RECOGNITION MODEL TRAINER
=============================================================
  
  HOW TO USE:
  -----------
  1. Create a folder called "training_data" in this directory
  2. Inside "training_data", create one sub-folder per person
     with their name as the folder name.
     
     Example structure:
       training_data/
         Anya/
           photo1.jpg
           photo2.jpg
           photo3.jpg
         Mom/
           photo1.jpg
           photo2.jpg
         Dad/
           photo1.jpg
           photo2.jpg
         Unknown/
           random_face1.jpg
           random_face2.jpg

  3. Put at least 5-10 photos per person (more = better)
     - Clear face photos (front-facing works best)
     - Different angles and lighting help
     - JPG, PNG, JPEG formats supported

  4. Run:  python train_model.py

  5. The new model is saved and the app will use it!
=============================================================
"""

import os
import sys
import cv2
import numpy as np
import pickle
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

# ============================================================
# Configuration
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
MODELS_DIR = os.path.join(STATIC_DIR, 'models')
TRAINING_DATA_DIR = os.path.join(BASE_DIR, 'training_data')

FACE_PROTO = os.path.join(MODELS_DIR, 'deploy.prototxt.txt')
FACE_MODEL = os.path.join(MODELS_DIR, 'res10_300x300_ssd_iter_140000.caffemodel')
FEATURE_MODEL = os.path.join(MODELS_DIR, 'openface.nn4.small2.v1.t7')
OUTPUT_FACE_MODEL = os.path.join(MODELS_DIR, 'celebrity_identity_model.pkl')

SUPPORTED_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
CONFIDENCE_THRESHOLD = 0.5  # minimum face detection confidence


def print_banner():
    print("\n" + "=" * 60)
    print("  FACE RECOGNITION MODEL TRAINER")
    print("=" * 60)


def load_models():
    """Load face detection and feature extraction models."""
    print("\n[1/4] Loading DNN models...")
    face_detector = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
    feature_extractor = cv2.dnn.readNetFromTorch(FEATURE_MODEL)
    print("  ✓ Face detector loaded")
    print("  ✓ Feature extractor loaded")
    return face_detector, feature_extractor


def extract_features(image_path, face_detector, feature_extractor):
    """Extract 128-dim face feature vector from an image."""
    img = cv2.imread(image_path)
    if img is None:
        return None

    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1, (300, 300), (104, 177, 123), swapRB=False, crop=False)
    face_detector.setInput(blob)
    detections = face_detector.forward()

    best_confidence = 0
    best_face_roi = None

    # Find the face with highest confidence
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > CONFIDENCE_THRESHOLD and confidence > best_confidence:
            best_confidence = confidence
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            startx, starty, endx, endy = box.astype(int)
            # Clamp to image bounds
            startx, starty = max(0, startx), max(0, starty)
            endx, endy = min(w, endx), min(h, endy)
            best_face_roi = img[starty:endy, startx:endx]

    if best_face_roi is None or best_face_roi.size == 0:
        return None

    # Extract features
    face_blob = cv2.dnn.blobFromImage(best_face_roi, 1 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=True)
    feature_extractor.setInput(face_blob)
    vectors = feature_extractor.forward()
    return vectors.flatten()


def collect_training_data(face_detector, feature_extractor):
    """Scan training_data folder and extract features for each person."""
    print("\n[2/4] Collecting training data...")

    if not os.path.exists(TRAINING_DATA_DIR):
        os.makedirs(TRAINING_DATA_DIR)
        print(f"\n  ✗ Created empty folder: {TRAINING_DATA_DIR}")
        print(f"    Please add person sub-folders with face images.")
        print(f"    See instructions at the top of this script.")
        sys.exit(1)

    persons = [d for d in os.listdir(TRAINING_DATA_DIR)
               if os.path.isdir(os.path.join(TRAINING_DATA_DIR, d))]

    if len(persons) < 2:
        print(f"\n  ✗ Need at least 2 person folders, found {len(persons)}")
        print(f"    Add sub-folders with face images in: {TRAINING_DATA_DIR}")
        sys.exit(1)

    print(f"  Found {len(persons)} persons: {', '.join(persons)}")

    features = []
    labels = []
    skipped = 0

    for person in persons:
        person_dir = os.path.join(TRAINING_DATA_DIR, person)
        images = [f for f in os.listdir(person_dir)
                  if os.path.splitext(f)[1].lower() in SUPPORTED_EXT]

        if len(images) == 0:
            print(f"  ⚠ No images found for '{person}', skipping")
            continue

        person_count = 0
        for img_name in images:
            img_path = os.path.join(person_dir, img_name)
            feat = extract_features(img_path, face_detector, feature_extractor)
            if feat is not None:
                features.append(feat)
                labels.append(person)
                person_count += 1
            else:
                skipped += 1

        print(f"  ✓ {person}: {person_count} faces extracted from {len(images)} images")

    if skipped > 0:
        print(f"  ⚠ Skipped {skipped} images (no face detected)")

    if len(features) < 4:
        print(f"\n  ✗ Not enough face samples ({len(features)}). Need at least 4.")
        print(f"    Add more clear face photos.")
        sys.exit(1)

    return np.array(features), np.array(labels)


def train_model(features, labels):
    """Train a VotingClassifier (same architecture as the original)."""
    print(f"\n[3/4] Training model on {len(features)} samples, {len(set(labels))} classes...")

    # Same model architecture as the original project
    # Using n_jobs=-1 to use all CPU cores where possible to speed up training
    svc = SVC(kernel='rbf', probability=True, C=10, gamma='scale')
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    dt = BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=50,
        random_state=42,
        n_jobs=-1
    )

    model = VotingClassifier(
        estimators=[('svc', svc), ('rf', rf), ('dt', dt)],
        voting='soft',
        n_jobs=-1
    )

    model.fit(features, labels)

    # -------------------------------------------------------------
    # Cross-validation is disabled to drastically speed up training! 
    # With 10,000 samples, cross_val_score repeats the entire training process 5 extra times.
    # -------------------------------------------------------------
    # if len(features) >= 10:
    #     scores = cross_val_score(model, features, labels, cv=min(5, len(set(labels))), scoring='accuracy')
    #     print(f"  ✓ Cross-validation accuracy: {scores.mean():.1%} (+/- {scores.std():.1%})")
    
    print(f"  ✓ Model trained successfully!")
    print(f"  ✓ Classes: {list(model.classes_)}")
    return model


def save_model(model):
    """Save the trained model."""
    print(f"\n[4/4] Saving model...")

    # Backup old model
    if os.path.exists(OUTPUT_FACE_MODEL):
        backup_path = OUTPUT_FACE_MODEL + '.backup'
        if not os.path.exists(backup_path):
            import shutil
            shutil.copy2(OUTPUT_FACE_MODEL, backup_path)
            print(f"  ✓ Old model backed up to: {os.path.basename(backup_path)}")

    with open(OUTPUT_FACE_MODEL, 'wb') as f:
        pickle.dump(model, f)

    print(f"  ✓ New model saved to: {os.path.basename(OUTPUT_FACE_MODEL)}")


def main():
    print_banner()
    face_detector, feature_extractor = load_models()
    features, labels = collect_training_data(face_detector, feature_extractor)
    model = train_model(features, labels)
    save_model(model)

    print("\n" + "=" * 60)
    print("  DONE! Restart the server to use the new model.")
    print("  python manage.py runserver")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
