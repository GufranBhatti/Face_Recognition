# Face Recognition Attendance Microservice

This project is a RESTful API microservice built using Python and Flask. Its primary purpose is to handle facial registration and matching (authentication) for an attendance system. The application incorporates state-of-the-art **Face Recognition** using `dlib` and **Liveness Detection** (Anti-Spoofing) via ONNX Runtime to prevent unauthorized access using photos or screens.

## 🌟 Key Concepts

### Face Recognition
The microservice uses the `face_recognition` library to extract a 128-dimensional embedding (face encoding) from images. During the matching phase, the Euclidean distance between the stored encoding and the uploaded encoding is calculated. A match is confirmed if the distance is below a strict tolerance (e.g., 0.45).

### Presentation Attack Detection (Liveness / Anti-Spoofing)
To ensure the face is a real live human being, the service uses **Silent-Face-Anti-Spoofing** via `onnxruntime`. The module runs two PyTorch models exported to ONNX (`MiniFASNetV1SE` and `MiniFASNetV2`). When an image is received, the system crops the face region, preprocesses it, and scores it across three classes: *Real Face*, *Printed Paper*, and *Screen Photo*. A fake face automatically blocks the registration or matching process with a `400 Bad Request`.

## 📂 Project Structure

- **`app.py`**: The main Flask application entry point. It sets up CORS, defines API routing endpoints, and bridges HTTP requests to the core logic.
- **`face_utils.py`**: Contains the core logic for the project. Handles loading images, ensuring quality (size/multiple faces), integrating the liveness checks, computing face encodings, saving users, and comparing faces.
- **`liveness_utils.py`**: Handles the anti-spoofing mechanism. Preloads the ONNX models into memory, crops facial bounding boxes with specific scale factors, and evaluates the liveness score of a face.
- **`anti_spoof_models/`**: Directory storing the pre-trained `.onnx` neural network models used by `liveness_utils.py`.
- **`static/`**:
  - `uploads/`: Temporary storage for images coming in from API requests.
  - `faces/`: Persistent storage. Each registered user has their own subdirectory containing their canonical `image.jpg`.
- **`test_api.py` & `test_spoof.py`**: Python scripts created to locally test all API endpoints and verify that spoofing detection works.

## 🚀 API Endpoints

### 1. `GET /`
Health check to confirm the API is running.

### 2. `POST /api/register`
Registers a new user face into the system.
- **Body (`form-data`)**: `userid` (String) and `image` (File).
- **Process**: Validates there is exactly one face, checks for liveness, and saves the image to `static/faces/{userid}/image.jpg`.

### 3. `POST /api/match`
Authenticates a user by matching their uploaded photo against the registered database.
- **Body (`form-data`)**: `userid` (String) and `image` (File).
- **Process**: Verifies liveness first, extracts the face encoding from the upload, and compares it strictly against the stored encoding. Returns `{"match": true}` or `false`.

### 4. `GET /api/is_enrolled`
Checks whether a user is already registered in the system.
- **Query Params**: `employeeid`
- **Process**: Checks the filesystem to see if the user's directory exists. Returns `{"is_enrolled": true/false}`.

## 🛠️ Installation & Setup

1. **Create and Activate a Virtual Environment**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate   # On Windows
   source venv/bin/activate  # On Mac/Linux
   ```

2. **Install Dependencies**
   The project requires `dlib` which can be tricky to compile on Windows. Use the provided `.whl` file if on Python 3.12:
   ```bash
   pip install dlib-19.24.99-cp312-cp312-win_amd64.whl
   pip install -r requirements.txt
   ```

3. **Run the Microservice**
   Start the Flask development server:
   ```bash
   python app.py
   ```
   The service will be available at `http://127.0.0.1:5000`.

## 🧪 Testing

You can use the provided Python scripts to test the API locally without needing Postman or a frontend application:
- Run `python test_api.py` (ensure you place a valid `sample_face.jpg` in the root) to test registration, enrollment checks, and matching.
- Run `python test_spoof.py` (ensure you place a fake `fake_face.jpg` in the root) to test that the liveness model correctly catches screen photos and printouts.
