from pathlib import Path
from datetime import datetime
from collections import deque
import csv
import time

import cv2
import numpy as np
import mediapipe as mp
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================
# PATHS
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

FACE_LANDMARKER_PATH = MODEL_DIR / "face_landmarker.task"
GENDER_MODEL_PATH = MODEL_DIR / "gender_classifier.pt"
AGE_MODEL_PATH = MODEL_DIR / "age_group_classifier.pt"


# =========================
# SESSION / LOG
# =========================

SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH = LOG_DIR / f"tracking_session_{SESSION_ID}.csv"

LOG_HEADERS = [
    "session_id",
    "visitor_id",
    "timestamp",
    "face_detected",
    "raw_orientation",
    "stabilized_orientation",
    "orientation_confidence",
    "facing_seconds",
    "interest_level",
    "age_group",
    "age_confidence",
    "gender",
    "gender_confidence",
]


# =========================
# CONFIG
# =========================

VISITOR_ABSENCE_TIMEOUT = 8.0
DEMO_INTERVAL_SECONDS = 1.0
LOG_INTERVAL_SECONDS = 1.0

# Son testte iyi çalışan esnek eşikler
NOSE_OFFSET_THRESHOLD = 0.40
CHEEK_RATIO_THRESHOLD = 0.22
MOUTH_OFFSET_THRESHOLD = 0.40

# Tahmin titremesini azaltır
SMOOTHING_WINDOW = 15


SELECTED_LANDMARKS = {
    "forehead": 10,
    "nose_tip": 1,
    "nose_bridge": 168,
    "chin": 152,

    "left_eye_outer": 33,
    "left_eye_inner": 133,
    "right_eye_inner": 362,
    "right_eye_outer": 263,

    "mouth_left": 61,
    "mouth_right": 291,
    "mouth_top": 13,
    "mouth_bottom": 14,

    "left_cheek": 234,
    "right_cheek": 454,
}


# =========================
# DEVICE
# =========================

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


DEVICE = get_device()


# =========================
# MEDIAPIPE
# =========================

def create_landmarker():
    if not FACE_LANDMARKER_PATH.exists():
        raise FileNotFoundError(f"Face landmarker model not found: {FACE_LANDMARKER_PATH}")

    base_options = python.BaseOptions(model_asset_path=str(FACE_LANDMARKER_PATH))

    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1
    )

    return vision.FaceLandmarker.create_from_options(options)


# =========================
# DEMOGRAPHIC CLASSIFIERS
# =========================

def safe_torch_load(model_path, device):
    try:
        return torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(model_path, map_location=device)


def load_classifier(model_path):
    if not model_path.exists():
        raise FileNotFoundError(f"Classifier model not found: {model_path}")

    checkpoint = safe_torch_load(model_path, DEVICE)

    class_names = checkpoint["class_names"]
    num_classes = checkpoint.get("num_classes", len(class_names))

    model = mobilenet_v3_small(weights=None)

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, num_classes)

    model.load_state_dict(checkpoint["state_dict"])
    model.to(DEVICE)
    model.eval()

    image_size = checkpoint.get("image_size", 224)

    mean = checkpoint.get("normalization_mean", [0.485, 0.456, 0.406])
    std = checkpoint.get("normalization_std", [0.229, 0.224, 0.225])

    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return {
        "model": model,
        "class_names": class_names,
        "preprocess": preprocess,
    }


def predict_classifier(bundle, face_crop_bgr):
    if face_crop_bgr is None or face_crop_bgr.size == 0:
        return "unknown", 0.0

    try:
        image_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        tensor = bundle["preprocess"](pil_image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = bundle["model"](tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, predicted_idx = torch.max(probs, dim=1)

        predicted_idx = int(predicted_idx.item())
        confidence = float(confidence.item())

        class_names = bundle["class_names"]
        label = class_names[predicted_idx]

        return label, confidence

    except Exception:
        return "unknown", 0.0


# =========================
# LANDMARK RULE ORIENTATION
# =========================

def estimate_front_facing_from_landmarks(landmarks):
    """
    Landmark simetrisine göre yüz kameraya dönük mü karar verir.

    Mantık:
    - Burun göz merkezine yakın mı?
    - Burun yanak merkezine yakın mı?
    - Ağız merkezi göz merkezine yakın mı?
    - Sağ/sol yanak mesafeleri aşırı bozulmamış mı?

    4 kontrolden en az 2'si geçerse front_facing kabul ediyoruz.
    Bu sayede hafif/orta kafa dönüşleri hâlâ ilgili sayılır,
    tam yan profil ise not_facing olur.
    """

    nose = landmarks[1]

    left_eye_outer = landmarks[33]
    right_eye_outer = landmarks[263]

    mouth_left = landmarks[61]
    mouth_right = landmarks[291]

    left_cheek = landmarks[234]
    right_cheek = landmarks[454]

    eye_center_x = (left_eye_outer.x + right_eye_outer.x) / 2
    eye_width = abs(right_eye_outer.x - left_eye_outer.x) + 1e-6

    cheek_center_x = (left_cheek.x + right_cheek.x) / 2
    cheek_width = abs(right_cheek.x - left_cheek.x) + 1e-6

    mouth_center_x = (mouth_left.x + mouth_right.x) / 2

    nose_offset_eye = abs(nose.x - eye_center_x) / eye_width
    nose_offset_cheek = abs(nose.x - cheek_center_x) / cheek_width
    mouth_offset = abs(mouth_center_x - eye_center_x) / eye_width

    left_cheek_distance = abs(nose.x - left_cheek.x)
    right_cheek_distance = abs(right_cheek.x - nose.x)

    max_cheek_distance = max(left_cheek_distance, right_cheek_distance) + 1e-6
    min_cheek_distance = min(left_cheek_distance, right_cheek_distance)

    cheek_ratio = min_cheek_distance / max_cheek_distance

    checks = [
        nose_offset_eye < NOSE_OFFSET_THRESHOLD,
        nose_offset_cheek < NOSE_OFFSET_THRESHOLD,
        mouth_offset < MOUTH_OFFSET_THRESHOLD,
        cheek_ratio > CHEEK_RATIO_THRESHOLD,
    ]

    passed_checks = sum(checks)

    is_front = passed_checks >= 2

    if is_front:
        confidence = passed_checks / 4
    else:
        confidence = 1 - (passed_checks / 4)

    debug_info = {
        "nose_offset_eye": nose_offset_eye,
        "nose_offset_cheek": nose_offset_cheek,
        "mouth_offset": mouth_offset,
        "cheek_ratio": cheek_ratio,
        "passed_checks": passed_checks,
    }

    return is_front, confidence, debug_info


def most_common_prediction(predictions):
    if not predictions:
        return "not_facing"

    front_count = sum(1 for p in predictions if p == "front_facing")
    not_count = sum(1 for p in predictions if p == "not_facing")

    if front_count >= not_count:
        return "front_facing"

    return "not_facing"


def get_interest_level(facing_seconds):
    if facing_seconds <= 0:
        return "uninterested"
    if facing_seconds < 3:
        return "low_interest"
    if facing_seconds < 6:
        return "medium_interest"
    return "high_interest"


# =========================
# DRAWING / CROP
# =========================

def draw_selected_landmarks(frame, landmarks):
    height, width = frame.shape[:2]

    for name, index in SELECTED_LANDMARKS.items():
        lm = landmarks[index]

        x = int(lm.x * width)
        y = int(lm.y * height)

        cv2.circle(frame, (x, y), 3, (0, 255, 180), -1)


def get_face_crop(frame, landmarks, padding_ratio=0.25):
    height, width = frame.shape[:2]

    xs = [int(lm.x * width) for lm in landmarks]
    ys = [int(lm.y * height) for lm in landmarks]

    x_min = max(0, min(xs))
    x_max = min(width, max(xs))
    y_min = max(0, min(ys))
    y_max = min(height, max(ys))

    box_w = x_max - x_min
    box_h = y_max - y_min

    if box_w <= 0 or box_h <= 0:
        return None, None

    pad_x = int(box_w * padding_ratio)
    pad_y = int(box_h * padding_ratio)

    x1 = max(0, x_min - pad_x)
    y1 = max(0, y_min - pad_y)
    x2 = min(width, x_max + pad_x)
    y2 = min(height, y_max + pad_y)

    crop = frame[y1:y2, x1:x2]

    return crop, (x1, y1, x2, y2)


def draw_text(frame, text, y, color=(230, 230, 230), scale=0.65, thickness=2):
    cv2.putText(
        frame,
        text,
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness
    )


def draw_face_box(frame, box):
    if box is None:
        return

    x1, y1, x2, y2 = box

    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 220, 255), 2)


# =========================
# MAIN
# =========================

def main():
    print(f"Device: {DEVICE}")

    landmarker = create_landmarker()
    gender_bundle = load_classifier(GENDER_MODEL_PATH)
    age_bundle = load_classifier(AGE_MODEL_PATH)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Camera could not be opened.")

    with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(LOG_HEADERS)

    visitor_counter = 0
    active_visitor_id = None
    last_seen_time = None

    recent_predictions = deque(maxlen=SMOOTHING_WINDOW)

    front_start_time = None
    facing_seconds = 0.0
    interest_level = "uninterested"

    age_group = "unknown"
    age_confidence = 0.0
    gender = "unknown"
    gender_confidence = 0.0

    last_demo_time = 0.0
    last_log_time = 0.0

    print("Tracking app with landmark-rule orientation started.")
    print(f"Logging to: {LOG_PATH}")
    print("Press q to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Frame could not be read.")
            break

        now = time.time()

        frame = cv2.flip(frame, 1)

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb = np.ascontiguousarray(image_rgb)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image_rgb
        )

        result = landmarker.detect(mp_image)

        face_detected = bool(result.face_landmarks)
        landmarks = None
        face_crop = None
        face_box = None
        debug_info = None

        if face_detected:
            landmarks = result.face_landmarks[0]
            face_crop, face_box = get_face_crop(frame, landmarks)

            if active_visitor_id is None:
                visitor_counter += 1
                active_visitor_id = f"visitor_{visitor_counter:03d}"
                print(f"New visitor started: {active_visitor_id}")

                front_start_time = None
                facing_seconds = 0.0
                interest_level = "uninterested"

                age_group = "unknown"
                age_confidence = 0.0
                gender = "unknown"
                gender_confidence = 0.0

                recent_predictions.clear()

            last_seen_time = now

        else:
            if active_visitor_id is not None and last_seen_time is not None:
                absence_duration = now - last_seen_time

                if absence_duration >= VISITOR_ABSENCE_TIMEOUT:
                    print(f"Visitor ended: {active_visitor_id}")

                    active_visitor_id = None
                    last_seen_time = None

                    front_start_time = None
                    facing_seconds = 0.0
                    interest_level = "uninterested"

                    age_group = "unknown"
                    age_confidence = 0.0
                    gender = "unknown"
                    gender_confidence = 0.0

                    recent_predictions.clear()

        # Orientation
        if not face_detected:
            raw_orientation = "not_facing"
            stable_orientation = "not_facing"
            orientation_confidence = 1.0
            recent_predictions.clear()
        else:
            is_front, orientation_confidence, debug_info = estimate_front_facing_from_landmarks(landmarks)
            raw_orientation = "front_facing" if is_front else "not_facing"

            recent_predictions.append(raw_orientation)
            stable_orientation = most_common_prediction(recent_predictions)

        # Interest timer
        if active_visitor_id is not None and stable_orientation == "front_facing":
            if front_start_time is None:
                front_start_time = now

            facing_seconds = now - front_start_time
            interest_level = get_interest_level(facing_seconds)

        else:
            front_start_time = None
            facing_seconds = 0.0
            interest_level = "uninterested"

        # Demographic prediction
        if (
            active_visitor_id is not None
            and face_detected
            and face_crop is not None
            and now - last_demo_time >= DEMO_INTERVAL_SECONDS
        ):
            age_group, age_confidence = predict_classifier(age_bundle, face_crop)
            gender, gender_confidence = predict_classifier(gender_bundle, face_crop)
            last_demo_time = now

        # Draw
        if landmarks is not None:
            draw_selected_landmarks(frame, landmarks)

        draw_face_box(frame, face_box)

        color = (0, 255, 120) if stable_orientation == "front_facing" else (0, 0, 255)

        draw_text(frame, f"Session: {SESSION_ID}", 30)
        draw_text(frame, f"Visitor: {active_visitor_id if active_visitor_id else 'none'}", 60)
        draw_text(frame, f"Face detected: {int(face_detected)}", 90)
        draw_text(frame, f"Raw orientation: {raw_orientation}", 120)
        draw_text(frame, f"Stable orientation: {stable_orientation}", 150, color=color)
        draw_text(frame, f"Facing seconds: {facing_seconds:.2f}", 180, color=(80, 220, 255))
        draw_text(frame, f"Interest: {interest_level}", 210, color=color)
        draw_text(frame, f"Age group: {age_group} ({age_confidence:.2f})", 245)
        draw_text(frame, f"Gender: {gender} ({gender_confidence:.2f})", 275)

        if debug_info is not None:
            y = 315
            for key, value in debug_info.items():
                if isinstance(value, float):
                    text = f"{key}: {value:.3f}"
                else:
                    text = f"{key}: {value}"

                draw_text(frame, text, y, color=(200, 200, 200), scale=0.45, thickness=1)
                y += 22

        draw_text(frame, "q: quit", frame.shape[0] - 20, color=(230, 230, 230), scale=0.55)

        # Log every second
        if now - last_log_time >= LOG_INTERVAL_SECONDS:
            timestamp = datetime.now().isoformat(timespec="seconds")

            row = [
                SESSION_ID,
                active_visitor_id if active_visitor_id is not None else "none",
                timestamp,
                int(face_detected),
                raw_orientation,
                stable_orientation,
                round(float(orientation_confidence), 4),
                round(float(facing_seconds), 2),
                interest_level,
                age_group,
                round(float(age_confidence), 4),
                gender,
                round(float(gender_confidence), 4),
            ]

            with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row)

            last_log_time = now

        cv2.imshow("Tracking App - Landmark Rule Orientation", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

    print("Tracking app with landmark-rule orientation closed.")
    print(f"Session log saved to: {LOG_PATH}")


if __name__ == "__main__":
    main()
