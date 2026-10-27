import cv2

from app.viewer_detection.yunet_detector import YuNetDetector
from app.viewer_detection.proximity_estimator import (
    ProximityEstimator,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = (
    "models/face_detection_yunet_2026may.onnx"
)

WINDOW_NAME = "SafeSight - Proximity Estimation"

# Put your actual distance from the camera here.
CALIBRATION_DISTANCE_CM = 60.0


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
# PROXIMITY ESTIMATOR
# ============================================================

estimator = ProximityEstimator(
    reference_distance_cm=CALIBRATION_DISTANCE_CM
)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open webcam."
    )


print()
print("=" * 60)
print("SafeSight - Proximity Estimation")
print("=" * 60)
print()
print(
    "Sit at a known distance from the camera."
)
print(
    f"Current calibration distance: "
    f"{CALIBRATION_DISTANCE_CM:.0f} cm"
)
print()
print("Press C to calibrate.")
print("Press Q to quit.")
print("=" * 60)


calibrated = False


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

    # --------------------------------------------------------
    # DETECT FACES
    # --------------------------------------------------------

    faces = detector.detect(frame)

    # --------------------------------------------------------
    # PROCESS FACES
    # --------------------------------------------------------

    for index, face in enumerate(faces):

        x, y, w, h = face["bbox"]

        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)

        face_width = w

        # ----------------------------------------------------
        # DISTANCE
        # ----------------------------------------------------

        result = estimator.estimate(
            face_width
        )

        distance = result["distance_cm"]
        proximity = result["proximity"]

        # ----------------------------------------------------
        # BOX
        # ----------------------------------------------------

        if proximity == "VERY CLOSE":

            box_color = (
                0,
                0,
                255,
            )

        elif proximity == "CLOSE":

            box_color = (
                0,
                165,
                255,
            )

        elif proximity == "MEDIUM":

            box_color = (
                0,
                255,
                255,
            )

        elif proximity == "FAR":

            box_color = (
                0,
                255,
                0,
            )

        else:

            box_color = (
                255,
                255,
                255,
            )

        cv2.rectangle(
            display,
            (x, y),
            (x + w, y + h),
            box_color,
            2,
        )

        # ----------------------------------------------------
        # PERSON LABEL
        # ----------------------------------------------------

        cv2.putText(
            display,
            f"Person {index + 1}",
            (x, y - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            box_color,
            2,
        )

        # ----------------------------------------------------
        # PROXIMITY
        # ----------------------------------------------------

        cv2.putText(
            display,
            f"Proximity: {proximity}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            box_color,
            2,
        )

        # ----------------------------------------------------
        # DISTANCE
        # ----------------------------------------------------

        if distance is not None:

            cv2.putText(
                display,
                f"Distance: {distance:.1f} cm",
                (x, y + h + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

        # ----------------------------------------------------
        # FACE WIDTH
        # ----------------------------------------------------

        cv2.putText(
            display,
            f"Face width: {face_width}px",
            (x, y + h + 47),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )

    # --------------------------------------------------------
    # GLOBAL STATUS
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

    # --------------------------------------------------------
    # CALIBRATION STATUS
    # --------------------------------------------------------

    if calibrated:

        status = "CALIBRATED"

        status_color = (
            0,
            255,
            0,
        )

    else:

        status = "NOT CALIBRATED - Press C"

        status_color = (
            0,
            0,
            255,
        )

    cv2.putText(
        display,
        status,
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        status_color,
        2,
    )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    cv2.imshow(
        WINDOW_NAME,
        display
    )

    key = cv2.waitKey(1) & 0xFF

    # --------------------------------------------------------
    # CALIBRATION
    # --------------------------------------------------------

    if key == ord("c"):

        if len(faces) == 1:

            x, y, w, h = faces[0]["bbox"]

            estimator.calibrate(
                face_width_px=w,
                distance_cm=CALIBRATION_DISTANCE_CM,
            )

            calibrated = True

            print()
            print(
                "Calibration successful!"
            )
            print(
                f"Reference distance : "
                f"{CALIBRATION_DISTANCE_CM:.1f} cm"
            )
            print(
                f"Reference face width: "
                f"{w:.1f} px"
            )
            print()

        else:

            print()
            print(
                "Calibration requires exactly "
                "ONE detected face."
            )
            print()

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

print(
    "Proximity estimation stopped."
)