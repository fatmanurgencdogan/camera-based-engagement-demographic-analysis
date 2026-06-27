from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "face_landmarker.task"
REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


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


CONNECTIONS = [
    ("left_eye_outer", "left_eye_inner"),
    ("right_eye_inner", "right_eye_outer"),
    ("left_eye_inner", "nose_bridge"),
    ("right_eye_inner", "nose_bridge"),
    ("nose_bridge", "nose_tip"),
    ("nose_tip", "mouth_top"),
    ("mouth_left", "mouth_right"),
    ("mouth_top", "mouth_bottom"),
    ("forehead", "nose_bridge"),
    ("nose_tip", "chin"),
    ("left_cheek", "right_cheek"),
]


SHORT_LABELS = {
    "forehead": "forehead",
    "nose_tip": "nose",
    "nose_bridge": "bridge",
    "chin": "chin",
    "left_eye_outer": "L eye",
    "left_eye_inner": "L eye in",
    "right_eye_inner": "R eye in",
    "right_eye_outer": "R eye",
    "mouth_left": "mouth L",
    "mouth_right": "mouth R",
    "mouth_top": "mouth top",
    "mouth_bottom": "mouth bot",
    "left_cheek": "L cheek",
    "right_cheek": "R cheek",
}


def create_landmarker():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Face landmarker model not found: {MODEL_PATH}")

    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))

    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1
    )

    return vision.FaceLandmarker.create_from_options(options)


def landmark_to_pixel(landmark, width, height):
    x = int(landmark.x * width)
    y = int(landmark.y * height)
    return x, y


def draw_face_box(frame, landmarks):
    height, width = frame.shape[:2]

    xs = [int(lm.x * width) for lm in landmarks]
    ys = [int(lm.y * height) for lm in landmarks]

    x_min = max(0, min(xs))
    x_max = min(width, max(xs))
    y_min = max(0, min(ys))
    y_max = min(height, max(ys))

    cv2.rectangle(
        frame,
        (x_min, y_min),
        (x_max, y_max),
        (80, 220, 255),
        2
    )

    cv2.putText(
        frame,
        "Detected Face",
        (x_min, max(25, y_min - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (80, 220, 255),
        2
    )


def draw_selected_landmarks(frame, landmarks):
    height, width = frame.shape[:2]

    pixel_points = {}

    for name, index in SELECTED_LANDMARKS.items():
        landmark = landmarks[index]
        x, y = landmark_to_pixel(landmark, width, height)
        pixel_points[name] = (x, y)

    for start_name, end_name in CONNECTIONS:
        if start_name in pixel_points and end_name in pixel_points:
            cv2.line(
                frame,
                pixel_points[start_name],
                pixel_points[end_name],
                (80, 180, 255),
                1
            )

    for name, point in pixel_points.items():
        x, y = point

        cv2.circle(
            frame,
            (x, y),
            4,
            (0, 255, 180),
            -1
        )

        cv2.circle(
            frame,
            (x, y),
            7,
            (0, 255, 180),
            1
        )

        label = SHORT_LABELS.get(name, name)

        cv2.putText(
            frame,
            label,
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1
        )


def main():
    landmarker = create_landmarker()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Camera could not be opened.")

    print("Landmark debug started.")
    print("Press q to quit.")
    print("Press s to save screenshot.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Frame could not be read.")
            break

        frame = cv2.flip(frame, 1)

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb = np.ascontiguousarray(image_rgb)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image_rgb
        )

        result = landmarker.detect(mp_image)

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]

            draw_face_box(frame, landmarks)
            draw_selected_landmarks(frame, landmarks)

            cv2.putText(
                frame,
                f"Selected landmarks: {len(SELECTED_LANDMARKS)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 180),
                2
            )
        else:
            cv2.putText(
                frame,
                "No face detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        cv2.putText(
            frame,
            "q: quit | s: save screenshot",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (230, 230, 230),
            2
        )

        cv2.imshow("MediaPipe Landmark Debug", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("s"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = REPORT_DIR / f"landmark_debug_{timestamp}.png"
            cv2.imwrite(str(save_path), frame)
            print(f"Screenshot saved: {save_path}")

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

    print("Landmark debug closed.")


if __name__ == "__main__":
    main()
