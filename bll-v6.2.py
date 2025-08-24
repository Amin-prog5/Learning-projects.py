# =========================================
# Fast Embedding Extraction with DeepFace
# =========================================

from deepface import DeepFace
import os
import cv2
import pickle
import gc

dataset_path = r"your paaath "
classes = os.listdir(dataset_path)
print("Detected classes:", classes)

def load_and_resize(img_path, size=(160, 160)):
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.resize(img, size)
    return img

known_embeddings = []
known_names = []

for person in classes:
    person_path = os.path.join(dataset_path, person)
    for img_name in os.listdir(person_path)[:70]:  # Limit to 20 images per person
        img_path = os.path.join(person_path, img_name)
        try:
            img = load_and_resize(img_path)
            if img is not None:
                embedding = DeepFace.represent(img, model_name='SFace')[0]["embedding"]
                known_embeddings.append(embedding)
                known_names.append(person)
        except Exception as e:
            print("Face not detected:", img_path)
        gc.collect()

# Save embeddings
with open("known_faces.pkl", "wb") as f:
    pickle.dump({"embeddings": known_embeddings, "names": known_names}, f)

print("✅ Embeddings saved successfully!")
