import cv2

from app.viewer_detection.yunet_detector import (
    YuNetDetector,
)

from app.viewer_detection.proximity_estimator import (
    ProximityEstimator,
)

from app.viewer_detection.distance_filter import (
    DistanceFilter,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = (
    "models/face_detection_yunet_2026may.onnx"
)

REFERENCE_DISTANCE_CM = 60.0

WINDOW_NAME = (
    "SafeSight - Smoothed Proximity"
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
# PROXIMITY ESTIMATOR
# ============================================================

estimator = ProximityEstimator(
    reference_distance_cm=REFERENCE_DISTANCE_CM
)


# ============================================================
# DISTANCE FILTER
# ============================================================

distance_filter = DistanceFilter(
    window_size=7,
    alpha=0.20,
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
print("SafeSight - Smoothed Proximity")
print("=" * 60)
print()
print(
    "Sit at 60 cm and press C to calibrate."
)
print()
print("Press C = calibrate")
print("Press R = reset filter")
print("Press Q = quit")
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
    # USE ONLY ONE FACE FOR THIS TEST
    # --------------------------------------------------------

    if len(faces) == 1:

        face = faces[0]

        x, y, w, h = face["bbox"]

        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)

        face_width = w

        # ----------------------------------------------------
        # CALIBRATION
        # ----------------------------------------------------

        if not calibrated:

            cv2.rectangle(
                display,
                (x, y),
                (x + w, y + h),
                (0, 255, 255),
                2,
            )

            cv2.putText(
                display,
                "Press C to calibrate",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

        else:

            # ------------------------------------------------
            # RAW DISTANCE
            # ------------------------------------------------

            result = estimator.estimate(
                face_width
            )

            raw_distance = (
                result["distance_cm"]
            )

            # ------------------------------------------------
            # SMOOTHED DISTANCE
            # ------------------------------------------------

            smoothed_distance = (
                distance_filter.update(
                    raw_distance
                )
            )

            # ------------------------------------------------
            # PROXIMITY
            # ------------------------------------------------

            proximity = (
                estimator.classify(
                    smoothed_distance
                )
            )

            # ------------------------------------------------
            # COLOR
            # ------------------------------------------------

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

            else:

                box_color = (
                    0,
                    255,
                    0,
                )

            # ------------------------------------------------
            # FACE BOX
            # ------------------------------------------------

            cv2.rectangle(
                display,
                (x, y),
                (x + w, y + h),
                box_color,
                2,
            )

            # ------------------------------------------------
            # RAW DISTANCE
            # ------------------------------------------------

            cv2.putText(
                display,
                f"Raw: {raw_distance:.1f} cm",
                (x, y - 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

            # ------------------------------------------------
            # SMOOTHED DISTANCE
            # ------------------------------------------------

            cv2.putText(
                display,
                f"Stable: {smoothed_distance:.1f} cm",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                box_color,
                2,
            )

            # ------------------------------------------------
            # PROXIMITY
            # ------------------------------------------------

            cv2.putText(
                display,
                f"Proximity: {proximity}",
                (x, y + h + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                box_color,
                2,
            )

            # ------------------------------------------------
            # FACE WIDTH
            # ------------------------------------------------

            cv2.putText(
                display,
                f"Face width: {face_width}px",
                (x, y + h + 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )

    elif len(faces) > 1:

        cv2.putText(
            display,
            "Calibration/test requires ONE face",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
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

        cv2.putText(
            display,
            "CALIBRATED",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    else:

        cv2.putText(
            display,
            "NOT CALIBRATED",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
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
    # CALIBRATE
    # --------------------------------------------------------

    if key == ord("c"):

        if len(faces) == 1:

            face = faces[0]

            _, _, width, _ = face["bbox"]

            estimator.calibrate(
                face_width_px=width,
                distance_cm=REFERENCE_DISTANCE_CM,
            )

            distance_filter.reset()

            calibrated = True

            print()
            print(
                "Calibration successful!"
            )
            print(
                f"Reference distance: "
                f"{REFERENCE_DISTANCE_CM:.1f} cm"
            )
            print(
                f"Reference face width: "
                f"{width:.1f} px"
            )
            print()

        else:

            print()
            print(
                "Calibration requires exactly "
                "ONE face."
            )
            print()

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    if key == ord("r"):

        distance_filter.reset()

        print(
            "Distance filter reset."
        )

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
    "Smoothed proximity test stopped."
)