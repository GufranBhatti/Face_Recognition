import os
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import Tuple, Dict

class AntiSpoofing:
    def __init__(self, model_dir: str = "anti_spoof_models"):
        self.model_dir = Path(model_dir)
        self.sessions = {}
        self.input_size = (80, 80)
        self._load_models()

    def _load_models(self):
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        onnx_files = list(self.model_dir.glob("*.onnx"))
        if not onnx_files:
            raise FileNotFoundError(f"No ONNX models found in {self.model_dir}")

        for onnx_file in onnx_files:
            try:
                session = ort.InferenceSession(
                    str(onnx_file),
                    providers=['CPUExecutionProvider']
                )
                self.sessions[onnx_file.name] = session
            except Exception as e:
                print(f"[FAIL] Failed to load {onnx_file.name}: {e}")

        if not self.sessions:
            raise RuntimeError("No models loaded successfully")

    def _get_new_box(self, src_w: int, src_h: int, bbox: list, scale: float) -> Tuple[int, int, int, int]:
        x, y, box_w, box_h = bbox
        scale = min((src_h - 1) / box_h, min((src_w - 1) / box_w, scale))

        new_width = box_w * scale
        new_height = box_h * scale
        center_x = box_w / 2 + x
        center_y = box_h / 2 + y

        left_top_x = center_x - new_width / 2
        left_top_y = center_y - new_height / 2
        right_bottom_x = center_x + new_width / 2
        right_bottom_y = center_y + new_height / 2

        if left_top_x < 0:
            right_bottom_x -= left_top_x
            left_top_x = 0
        if left_top_y < 0:
            right_bottom_y -= left_top_y
            left_top_y = 0
        if right_bottom_x > src_w - 1:
            left_top_x -= right_bottom_x - src_w + 1
            right_bottom_x = src_w - 1
        if right_bottom_y > src_h - 1:
            left_top_y -= right_bottom_y - src_h + 1
            right_bottom_y = src_h - 1

        return int(left_top_x), int(left_top_y), int(right_bottom_x), int(right_bottom_y)

    def _preprocess_face(self, img_bgr: np.ndarray, bbox: list, scale: float) -> np.ndarray:
        src_h, src_w = img_bgr.shape[:2]
        left_top_x, left_top_y, right_bottom_x, right_bottom_y = self._get_new_box(
            src_w, src_h, bbox, scale
        )
        face_crop = img_bgr[left_top_y:right_bottom_y + 1, left_top_x:right_bottom_x + 1]
        face_resized = cv2.resize(face_crop, self.input_size)
        face_float = face_resized.astype(np.float32)
        face_chw = np.transpose(face_float, (2, 0, 1))
        face_batch = np.expand_dims(face_chw, axis=0)
        return face_batch

    def _parse_model_name(self, model_name: str) -> float:
        parts = model_name.split('_')
        if parts[0] == '4':
            return 4.0
        return float(parts[0])

    def predict(self, img_bgr: np.ndarray, bbox: list) -> Dict:
        predictions = np.zeros((1, 3))
        for model_name, session in self.sessions.items():
            scale = self._parse_model_name(model_name)
            input_data = self._preprocess_face(img_bgr, bbox, scale)
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            output = session.run([output_name], {input_name: input_data})[0]

            exp_output = np.exp(output - np.max(output, axis=1, keepdims=True))
            softmax_output = exp_output / np.sum(exp_output, axis=1, keepdims=True)
            predictions += softmax_output

        predictions = predictions / len(self.sessions)
        label = int(np.argmax(predictions[0]))
        scores = predictions[0]

        label_names = ["Paper Photo", "Real Face", "Screen Photo"]
        result = {
            "label": label,
            "label_text": label_names[label],
            "scores": {"paper": float(scores[0]), "real": float(scores[1]), "screen": float(scores[2])},
            "is_real": (label == 1),
            "confidence": float(scores[label]),
        }
        return result

# Singleton instance
anti_spoofing_engine = None

def check_liveness(image_path: str, face_location: tuple) -> tuple:
    """
    Checks if the face in the image is real or a spoof.
    face_location is (top, right, bottom, left) from face_recognition.
    Returns (is_real, message).
    """
    global anti_spoofing_engine
    if anti_spoofing_engine is None:
        anti_spoofing_engine = AntiSpoofing()

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return False, "Failed to read image for liveness detection."

    top, right, bottom, left = face_location
    bbox = [left, top, right - left, bottom - top]

    result = anti_spoofing_engine.predict(img_bgr, bbox)
    if result["is_real"]:
        return True, "Real face detected."
    else:
        return False, f"Spoofing detected ({result['label_text']} with {result['confidence']:.2f} confidence)."
