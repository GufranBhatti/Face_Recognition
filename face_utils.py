import face_recognition
import os
import cv2
FACES_BASE_DIR = "static/faces"

def is_blurry(image_path, threshold=100.0):
    image = cv2.imread(image_path)
    if image is None:
        return True
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap_var < threshold

def save_user_face(image_path, userid):
    """
    Saves the uploaded face image if it's clear and valid.
    """
    if is_blurry(image_path):
        return False, "Image is too blurry. Please upload a clearer photo."

    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)

    if not face_locations:
        return False, "No face detected."

    if len(face_locations) > 1:
        return False, "Multiple faces detected. Only one person allowed."

    top, right, bottom, left = face_locations[0]
    face_height = bottom - top
    face_width = right - left
    image_height, image_width = image.shape[:2]

    if face_height < image_height * 0.3 or face_width < image_width * 0.3:
        return False, "Face is too small or not clearly visible."

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
    unknown_encodings = face_recognition.face_encodings(unknown_image)

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