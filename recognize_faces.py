import cv2
import face_recognition
import pickle

# Load face encodings
print("🔄 Loading encodings...")
try:
    with open("encodings.pkl", "rb") as f:
        data = pickle.load(f)
    print(f"✅ Encodings loaded! Found {len(data['encodings'])} faces.")
except FileNotFoundError:
    print("❌ Error: encodings.pkl not found! Run encode_faces.py first.")
    exit()

# Open webcam
print("📷 Opening webcam...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

print("✅ Webcam opened successfully! Starting face recognition...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Failed to capture frame")
        break

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    print(f"👀 Detected {len(face_locations)} face(s) in the frame.")

    for encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance=0.4)
        face_distances = face_recognition.face_distance(data["encodings"], encoding)

        name = "Unknown"

        # ✅ Select the best match (smallest distance)
        if len(face_distances) > 0:
            best_match_index = min(range(len(face_distances)), key=lambda i: face_distances[i])

            if matches[best_match_index] and face_distances[best_match_index] < 0.4:  # Lower tolerance for better accuracy
                name = data["names"][best_match_index]

        print(f"🆔 Recognized: {name}")

        # Draw a rectangle and display name
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
