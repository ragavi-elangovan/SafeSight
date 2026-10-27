import cv2
import json
import os

from app.viewer_detection.yunet_detector import (
    YuNetDetector,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = (
    "models/face_detection_yunet_2026may.onnx"
)

CONFIG_PATH = (
    "data/config/proximity_config.json"
)

REFERENCE_DISTANCE_CM = 60.0

WINDOW_NAME = (
    "SafeSight - One Time Proximity Calibration"
)


# ============================================================
# LOAD YUNET
# ============================================================

print("Loading YuNet...")

detector = YuNetDetector(
    model_path=MODEL_PATH,
    input_size=(640, 480),
)

print("YuNet loaded.")


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open webcam."
    )


print()
print("=" * 65)
print("SafeSight - ONE TIME PROXIMITY CALIBRATION")
print("=" * 65)
print()
print(
    "Sit approximately 60 cm from the camera."
)
print(
    "Make sure exactly ONE face is visible."
)
print()
print("Press C to save calibration.")
print("Press Q to quit.")
print("=" * 65)


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print(
            "Failed to read camera frame."
        )
        break

    display = frame.copy()

    faces = detector.detect(frame)

    # --------------------------------------------------------
    # FACE STATUS
    # --------------------------------------------------------

    cv2.putText(
        display,
        f"Faces detected: {len(faces)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    if len(faces) == 1:

        face = faces[0]

        x, y, w, h = face["bbox"]

        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)

        cv2.rectangle(
            display,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            display,
            f"Face width: {w}px",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            display,
            "Press C to save calibration",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )

    else:

        cv2.putText(
            display,
            "Exactly ONE face required",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
        )

    cv2.putText(
        display,
        "Reference distance: 60 cm",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )

    cv2.imshow(
        WINDOW_NAME,
        display,
    )

    key = cv2.waitKey(1) & 0xFF

    # --------------------------------------------------------
    # SAVE CALIBRATION
    # --------------------------------------------------------

    if key == ord("c"):

        if len(faces) != 1:

            print()
            print(
                "Calibration failed."
            )
            print(
                "Exactly ONE face is required."
            )
            print()

            continue

        face = faces[0]

        _, _, face_width, _ = face["bbox"]

        face_width = float(
            face_width
        )

        # ----------------------------------------------------
        # FIXED PRIVACY THRESHOLDS
        # ----------------------------------------------------

        config = {
            "reference_distance_cm": (
                REFERENCE_DISTANCE_CM
            ),

            "reference_face_width_px": (
                face_width
            ),

            "very_close_threshold_cm": 40.0,

            "close_threshold_cm": 70.0,

            "medium_threshold_cm": 120.0,

            "far_threshold_cm": 120.0,

            "distance_filter_window": 7,

            "distance_filter_alpha": 0.20,
        }

        # ----------------------------------------------------
        # CREATE DIRECTORY
        # ----------------------------------------------------

        os.makedirs(
            os.path.dirname(CONFIG_PATH),
            exist_ok=True,
        )

        # ----------------------------------------------------
        # SAVE JSON
        # ----------------------------------------------------

        with open(
            CONFIG_PATH,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                config,
                file,
                indent=4,
            )

        print()
        print("=" * 65)
        print("CALIBRATION SAVED SUCCESSFULLY")
        print("=" * 65)
        print(
            f"Reference distance : "
            f"{REFERENCE_DISTANCE_CM:.1f} cm"
        )
        print(
            f"Reference face width: "
            f"{face_width:.1f} px"
        )
        print()
        print(
            f"Saved to: {CONFIG_PATH}"
        )
        print()
        print(
            "You will NOT need to calibrate"
        )
        print(
            "every time SafeSight starts."
        )
        print("=" * 65)

        break

    # --------------------------------------------------------
    # QUIT
    # --------------------------------------------------------

    if key == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()