# =========================================
# Face Recognition Project - Kaggle Dataset
# Colab Ready (Optimized with DeepFace)
# =========================================

# 1- Work with Google Colab
# Install dependencies
!pip install kaggle deepface opencv-python-headless matplotlib tensorflow

# Upload your Kaggle API token (kaggle.json)
from google.colab import files
files.upload()  # Upload kaggle.json here

# Set Kaggle environment
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

import os
os.environ['KAGGLE_CONFIG_DIR'] = "/root/.kaggle"

# Download and unzip the dataset
!kaggle datasets download -d hereisburak/pins-face-recognition -p /content/dataset --unzip

# Verify dataset
!ls /content/dataset

# 2- Read & Organize Data (with Resize + Memory Cleanup)
import cv2
import numpy as np
import gc
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

dataset_path = "/content/dataset/105_classes_pins_dataset"
classes = os.listdir(dataset_path)
print("Detected classes:", classes)

def load_and_resize(img_path, size=(160, 160)):
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.resize(img, size)
    return img

images = []
labels = []

for label, person in enumerate(classes):
    person_path = os.path.join(dataset_path, person)
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = load_and_resize(img_path)
        if img is not None:
            images.append(img)
            labels.append(label)
        gc.collect()

images = np.array(images)
labels = np.array(labels)

# Split data
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# 3- Preprocessing
X_train = X_train / 255.0
X_test = X_test / 255.0
y_train = to_categorical(y_train, num_classes=len(classes))
y_test = to_categorical(y_test, num_classes=len(classes))

# Optional: Data augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(rotation_range=20,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             horizontal_flip=True)
datagen.fit(X_train)

# 4- Choose Technique (DeepFace Embeddings)
from deepface import DeepFace
import pickle

known_embeddings = []
known_names = []

for person in classes:
    person_path = os.path.join(dataset_path, person)
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        try:
            img = load_and_resize(img_path)
            if img is not None:
                embedding = DeepFace.represent(img, model_name='Facenet')[0]["embedding"]
                known_embeddings.append(embedding)
                known_names.append(person)
        except Exception as e:
            print("Face not detected:", img_path)
        gc.collect()

# 5- Save embeddings
with open("known_faces.pkl", "wb") as f:
    pickle.dump({"embeddings": known_embeddings, "names": known_names}, f)
print("Embeddings saved successfully!")

# 6- Evaluate on Test Set (Memory-Safe)
from scipy.spatial.distance import euclidean

correct = 0
total = 0

for i, img in enumerate(X_test):
    img_uint8 = (img * 255).astype(np.uint8)
    try:
        embedding = DeepFace.represent(img_uint8, model_name='Facenet')[0]["embedding"]
        distances = [euclidean(embedding, known) for known in known_embeddings]
        best_match_index = np.argmin(distances)
        predicted_name = known_names[best_match_index]
        true_name = classes[np.argmax(y_test[i])]
        if predicted_name == true_name:
            correct += 1
        total += 1
    except:
        continue
    gc.collect()

accuracy = correct / total if total > 0 else 0
print("Test Accuracy:", accuracy)

# 7- Real-Time Predictions (Memory-Safe)
import cv2

# Load saved embeddings
with open("known_faces.pkl", "rb") as f:
    data = pickle.load(f)
known_embeddings = data["embeddings"]
known_names = data["names"]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        results = DeepFace.extract_faces(frame, detector_backend='opencv')
        for res in results:
            x, y, w, h = res["facial_area"].values()
            face_img = res["face"]
            embedding = DeepFace.represent(face_img, model_name='Facenet')[0]["embedding"]

            distances = [euclidean(embedding, known) for known in known_embeddings]
            best_match_index = np.argmin(distances)
            name = known_names[best_match_index] if distances[best_match_index] < 0.6 else "Unknown"

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        gc.collect()
    except Exception as e:
        print("Error:", e)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
