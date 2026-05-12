import face_recognition
import os
import cv2
from liveness_utils import check_liveness
FACES_BASE_DIR = "static/faces"

def is_blurry(image_path, threshold=80.0):
    image = cv2.imread(image_path)
    if image is None:
        return True

    # Convert to RGB because face_recognition works on RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get face location
    face_locations = face_recognition.face_locations(rgb_image)

    if not face_locations:
        return True  # Can't detect face, so treat as blurry or invalid

    # Use the first face only
    top, right, bottom, left = face_locations[0]

    # Crop the face region
    face_region = image[top:bottom, left:right]

    # Check blur on cropped face only
    gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    return lap_var < threshold

def save_user_face(image_path, userid):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)

    if not face_locations:
        return False, "No face detected."
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if len(face_locations) > 1:
        return False, "Multiple faces detected. Only one person allowed."

    # Check liveness
    is_real, message = check_liveness(image_path, face_locations[0])
    if not is_real:
        return False, message

    top, right, bottom, left = face_locations[0]
    face_height = bottom - top
    face_width = right - left
    image_height, image_width = image.shape[:2]

    if face_height < image_height * 0.3 or face_width < image_width * 0.3:
        return False, "Face is too small or not clearly visible."

    # NEW: Try encoding and reject if it fails
    face_encoding = face_recognition.face_encodings(image, known_face_locations=[face_locations[0]])
    if not face_encoding:
        return False, "Face is unclear or blurry. Try a clearer photo."

    user_dir = os.path.join(FACES_BASE_DIR, userid)
    os.makedirs(user_dir, exist_ok=True)
    saved_image_path = os.path.join(user_dir, "image.jpg")
    os.replace(image_path, saved_image_path)

    return True, "Face registered successfully."

def match_face(userid, image_path):
    """
    Compare the given image with the registered image of a specific user.
    """
    user_folder = os.path.join("static/faces", userid)
    if not os.path.exists(user_folder):
        return False, "User not found."

    stored_image_path = os.path.join(user_folder, "image.jpg")
    if not os.path.exists(stored_image_path):
        return False, "Registered image not found."

    unknown_image = face_recognition.load_image_file(image_path)
    unknown_locations = face_recognition.face_locations(unknown_image)

    if not unknown_locations:
        return False, "No face detected in uploaded image."

    # Check liveness
    is_real, message = check_liveness(image_path, unknown_locations[0])
    if not is_real:
        return False, message

    unknown_encodings = face_recognition.face_encodings(unknown_image, known_face_locations=[unknown_locations[0]])

    if not unknown_encodings:
        return False, "No face detected in uploaded image."

    unknown_encoding = unknown_encodings[0]

    known_image = face_recognition.load_image_file(stored_image_path)
    known_encodings = face_recognition.face_encodings(known_image)

    if not known_encodings:
        return False, "No face found in stored image."

    known_encoding = known_encodings[0]

    matches = face_recognition.compare_faces([known_encoding], unknown_encoding, tolerance=0.45)
    face_distances = face_recognition.face_distance([known_encoding], unknown_encoding)
    confidence = 1.0 - face_distances[0]

    if matches[0]:
        return True, f"Face matched with confidence: {confidence:.2f}"
    else:
        return False, "Face did not match."