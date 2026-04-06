#Utility functions for model loading, image preprocessing, prediction, and logging.
import json
import os
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime
import pandas as pd
import csv


#Root Helpers
def find_project_root(start_path: str) -> str:
    current = os.path.abspath(start_path)
    if os.path.isfile(current):
        current = os.path.dirname(current)

    for _ in range(10):
        if any(
            os.path.exists(os.path.join(current, name))
            for name in ["artifacts", "app", "src", "models"]
        ):
            return current

        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent

    return os.path.dirname(os.path.abspath(start_path))


def first_existing_path(paths: List[str], label: str) -> str:
    for path in paths:
        if os.path.exists(path):
            return path

    checked = "\n".join(paths)
    raise FileNotFoundError(f"{label} not found. Checked:\n{checked}")


BASE_DIR = find_project_root(__file__)


def candidate_model_paths(base_dir: str) -> List[str]:
    return [
        os.path.join(base_dir, "artifacts", "model3_mobilenetv2.keras"),
        os.path.join(base_dir, "artifacts", "models", "model3_mobilenetv2.keras"),
        os.path.join(base_dir, "models", "model3_mobilenetv2.keras"),
        os.path.join(base_dir, "model3_mobilenetv2.keras"),
        os.path.join(base_dir, "artifacts", "evaluation", "model3_mobilenetv2.keras"),
    ]


def candidate_classmap_paths(base_dir: str) -> List[str]:
    return [
        os.path.join(base_dir, "artifacts", "class_mapping.json"),
        os.path.join(base_dir, "class_mapping.json"),
        os.path.join(base_dir, "artifacts", "metadata", "class_mapping.json"),
    ]


MODEL_PATH = first_existing_path(candidate_model_paths(BASE_DIR), "Model file")
CLASS_MAP_PATH = first_existing_path(candidate_classmap_paths(BASE_DIR), "Class mapping file")


#Global Cache
_model = None
_idx_to_class = None
_input_height = 224
_input_width = 224


#Load Model and Class Map
def load_model_and_mapping():
    global _model, _idx_to_class, _input_height, _input_width

    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH)

        try:
            input_shape = _model.input_shape
            if isinstance(input_shape, tuple) and len(input_shape) == 4:
                if input_shape[1] is not None:
                    _input_height = int(input_shape[1])
                if input_shape[2] is not None:
                    _input_width = int(input_shape[2])
        except Exception:
            _input_height, _input_width = 224, 224

    if _idx_to_class is None:
        with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
            mapping = json.load(f)

        if "id_to_label" not in mapping:
            raise KeyError("class_mapping.json must contain 'id_to_label'.")

        _idx_to_class = {int(k): str(v) for k, v in mapping["id_to_label"].items()}

    return _model, _idx_to_class


#Image Quality Assessment
def variance_of_laplacian(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def estimate_brightness(gray: np.ndarray) -> float:
    return float(np.mean(gray))


def estimate_contrast(gray: np.ndarray) -> float:
    return float(np.std(gray))


def image_quality_report(bgr: np.ndarray) -> Dict[str, Any]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    blur_score = variance_of_laplacian(gray)
    brightness = estimate_brightness(gray)
    contrast = estimate_contrast(gray)

    return {
        "blur_score": blur_score,
        "brightness": brightness,
        "contrast": contrast,
        "is_blurry": blur_score < 35.0,
        "is_dark": brightness < 60.0,
        "is_low_contrast": contrast < 25.0,
    }


#Preprocessing 
def safe_clahe(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def pad_to_square(img: np.ndarray, fill_value: int = 0) -> np.ndarray:
    h, w = img.shape[:2]
    size = max(h, w)

    if len(img.shape) == 2:
        canvas = np.full((size, size), fill_value, dtype=img.dtype)
    else:
        canvas = np.full((size, size, img.shape[2]), fill_value, dtype=img.dtype)

    y_offset = (size - h) // 2
    x_offset = (size - w) // 2
    canvas[y_offset:y_offset + h, x_offset:x_offset + w] = img
    return canvas


def largest_foreground_crop(gray: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int], np.ndarray]:
    h, w = gray.shape[:2]

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, otsu_binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    otsu_inv = 255 - otsu_binary

    best_area = -1.0
    best_bbox = (0, 0, w, h)
    best_crop = gray.copy()
    best_mask = np.zeros_like(gray)

    kernel = np.ones((5, 5), np.uint8)

    for binary in [otsu_binary, otsu_inv]:
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            continue

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 0.02 * (h * w):
                continue

            x, y, cw, ch = cv2.boundingRect(contour)

            if cw < 0.10 * w or ch < 0.10 * h:
                continue

            if area > best_area:
                pad_x = int(cw * 0.18)
                pad_y = int(ch * 0.18)

                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(w, x + cw + pad_x)
                y2 = min(h, y + ch + pad_y)

                best_area = area
                best_bbox = (x1, y1, x2 - x1, y2 - y1)
                best_crop = gray[y1:y2, x1:x2]
                best_mask = cleaned

    return best_crop, best_bbox, best_mask


def preprocess_for_model(
    bgr: np.ndarray,
    input_size: Tuple[int, int],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = safe_clahe(gray)

    cropped, bbox, mask = largest_foreground_crop(gray)
    squared = pad_to_square(cropped, fill_value=0)

    resized = cv2.resize(
        squared,
        input_size,
        interpolation=cv2.INTER_AREA if squared.shape[0] > input_size[0] else cv2.INTER_CUBIC,
    )

    rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    rgb = rgb.astype(np.float32) / 255.0

    meta = {
        "bbox": bbox,
        "crop_shape": cropped.shape,
        "square_shape": squared.shape,
        "final_shape": rgb.shape,
        "mask_preview": mask,
        "gray_preview": gray,
        "cropped_preview": cropped,
        "model_path": MODEL_PATH,
        "class_map_path": CLASS_MAP_PATH,
    }
    return rgb, meta


#Test Time Augmentation (TTA)
def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    return cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def build_tta_batch(preprocessed_rgb: np.ndarray) -> np.ndarray:
    base = (preprocessed_rgb * 255).astype(np.uint8)

    brighter = np.clip(base.astype(np.float32) * 1.08, 0, 255).astype(np.uint8)
    darker = np.clip(base.astype(np.float32) * 0.92, 0, 255).astype(np.uint8)
    rot_left = rotate_image(base, 6)
    rot_right = rotate_image(base, -6)

    variants = [base, brighter, darker, rot_left, rot_right]
    batch = np.stack([v.astype(np.float32) / 255.0 for v in variants], axis=0)
    return batch


#Main Predict Function
def predict_frame_with_debug(bgr: np.ndarray) -> Dict[str, Any]:
    model, idx_to_class = load_model_and_mapping()

    quality = image_quality_report(bgr)

    preprocessed_rgb, meta = preprocess_for_model(
        bgr,
        input_size=(_input_height, _input_width),
    )

    batch = build_tta_batch(preprocessed_rgb)
    probs_batch = model.predict(batch, verbose=0)
    probs = np.mean(probs_batch, axis=0)

    pred_idx = int(np.argmax(probs))
    pred_label = idx_to_class[pred_idx]
    confidence = float(probs[pred_idx])

    return {
        "pred_label": pred_label,
        "confidence": confidence,
        "probs": probs,
        "quality": quality,
        "meta": meta,
        "preprocessed_rgb": preprocessed_rgb,
    }
    

def load_word_curriculum(path="app/data/word_curriculum.csv"):
    if not os.path.exists(path):
        return pd.DataFrame(columns=["word", "difficulty", "category"])
    df = pd.read_csv(path)
    df["word"] = df["word"].astype(str).str.upper().str.strip()
    df["difficulty"] = df["difficulty"].astype(str).str.lower().str.strip()
    df["category"] = df["category"].astype(str).str.lower().str.strip()
    return df


def get_random_word(word_df, difficulty=None):
    if word_df.empty:
        return None
    temp_df = word_df.copy()
    if difficulty and difficulty.lower() != "all":
        temp_df = temp_df[temp_df["difficulty"] == difficulty.lower()]
    if temp_df.empty:
        return None
    row = temp_df.sample(1).iloc[0]
    return {
        "word": row["word"],
        "difficulty": row["difficulty"],
        "category": row["category"]
    }


def ensure_word_log_exists(path="app/logs/word_attempts_log.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "target_word",
                "predicted_word",
                "difficulty",
                "category",
                "target_length",
                "correct_letters",
                "word_correct",
                "avg_confidence",
                "wrong_positions",
                "failed_letters"
            ])


def save_word_attempt(
    target_word,
    predicted_word,
    difficulty,
    category,
    confidences,
    path="app/logs/word_attempts_log.csv"
):
    ensure_word_log_exists(path)

    target_word = str(target_word).upper().strip()
    predicted_word = str(predicted_word).upper().strip()

    correct_letters = sum(
        1 for t, p in zip(target_word, predicted_word) if t == p
    )

    word_correct = int(target_word == predicted_word)

    wrong_positions = []
    failed_letters = []

    for i, (t, p) in enumerate(zip(target_word, predicted_word), start=1):
        if t != p:
            wrong_positions.append(str(i))
            failed_letters.append(f"{t}->{p}")

    avg_confidence = round(sum(confidences) / len(confidences), 4) if confidences else 0.0

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            target_word,
            predicted_word,
            difficulty,
            category,
            len(target_word),
            correct_letters,
            word_correct,
            avg_confidence,
            ",".join(wrong_positions),
            ",".join(failed_letters)
        ])