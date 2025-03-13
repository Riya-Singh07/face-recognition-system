from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
import cv2
import face_recognition
import pickle

app = FastAPI()

# Load face encodings
try:
    with open("encodings.pkl", "rb") as f:
        data = pickle.load(f)
    print(f" Encodings loaded! Found {len(data['encodings'])} faces.")
except FileNotFoundError:
    print(" Error: encodings.pkl not found! Run encode_faces.py first.")
    exit()

# Open webcam
cap = cv2.VideoCapture(0)

# Define the square area for recognition (centered, 200x200 pixels)
FRAME_WIDTH = int(cap.get(3))  # Get frame width
FRAME_HEIGHT = int(cap.get(4))  # Get frame height
SQUARE_SIZE = 200

CENTER_X = FRAME_WIDTH // 2
CENTER_Y = FRAME_HEIGHT // 2

SQUARE_TOP_LEFT = (CENTER_X - SQUARE_SIZE // 2, CENTER_Y - SQUARE_SIZE // 2)
SQUARE_BOTTOM_RIGHT = (CENTER_X + SQUARE_SIZE // 2, CENTER_Y + SQUARE_SIZE // 2)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        recognized_name = "Unknown"
        for encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance=0.5)
            name = "Unknown"

            if True in matches:
                matched_idxs = [i for i, match in enumerate(matches) if match]
                name = data["names"][matched_idxs[0]]

            # Get face coordinates
            top, right, bottom, left = face_location

            # Check if face is inside the defined square area
            if (left > SQUARE_TOP_LEFT[0] and right < SQUARE_BOTTOM_RIGHT[0] and
                top > SQUARE_TOP_LEFT[1] and bottom < SQUARE_BOTTOM_RIGHT[1]):
                recognized_name = name  # Recognize only when inside the square
                cv2.putText(frame, f"Recognized: {name}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw rectangle around detected face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw the fixed square area
        cv2.rectangle(frame, SQUARE_TOP_LEFT, SQUARE_BOTTOM_RIGHT, (255, 0, 0), 2)

        # Encode frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.get("/")
def home():
    return {"message": "Face Recognition API Running"}

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")
