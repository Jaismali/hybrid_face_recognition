import face_recognition
import pickle
import os
import numpy as np

# Path to the images folder
IMAGES_PATH = "images"

# Lists to store encodings and names
known_encodings = []
known_names = []

# Load each image and extract face encodings
for filename in os.listdir(IMAGES_PATH):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        image_path = os.path.join(IMAGES_PATH, filename)
        image = face_recognition.load_image_file(image_path)

        # Extract face encodings
        encodings = face_recognition.face_encodings(image)
        if encodings:  # Ensure a face is detected
            known_encodings.append(encodings[0])
            known_names.append(os.path.splitext(filename)[0])  # Store name without extension
            print(f"✅ Encoded: {filename}")
        else:
            print(f"⚠️ No face found in {filename}, skipping.")

# Store encodings in a dictionary
data = {"encodings": known_encodings, "names": known_names}

# Save as a .pkl file
with open("face_encodings.pkl", "wb") as f:
    pickle.dump(data, f)

print("\n✅ face_encodings.pkl has been created successfully!")