import os
import cv2
import face_recognition 
import pickle

# Set the dataset path
DATASET_PATH = r"C:\\Face Recognition\\Bollywood Actor Images"
ENCODINGS_FILE = "encodings.pkl"

known_encodings = []
known_names = []

# Loop through each actor's folder
for actor_name in os.listdir(DATASET_PATH):
    actor_path = os.path.join(DATASET_PATH, actor_name)
    
    # Skip if not a directory
    if not os.path.isdir(actor_path):
        continue
    
    print(f"Processing {actor_name}...")  # Debugging output

    # Loop through images inside the actor's folder
    for image_name in os.listdir(actor_path):
        image_path = os.path.join(actor_path, image_name)
        
        try:
            # Read image
            image = cv2.imread(image_path)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Detect face and encode
            face_locations = face_recognition.face_locations(rgb_image)
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

            for encoding in face_encodings:
                known_encodings.append(encoding)
                known_names.append(actor_name)  # Actor name is from folder name

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

# Save encodings
data = {"encodings": known_encodings, "names": known_names}
with open(ENCODINGS_FILE, "wb") as f:
    pickle.dump(data, f)

print("Encoding complete! Encoded", len(known_encodings), "faces.")
