import cv2

from app.viewer_detection.yunet_detector import (
    YuNetDetector,
)

from app.viewer_detection.pose_tracker import (
    MultiPersonPoseTracker,
)

from app.viewer_detection.multi_person_proximity import (
    MultiPersonProximity,
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

WINDOW_NAME = (
    "SafeSight - Automatic Multi Person Proximity"
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
# LOAD TRACKER
# ============================================================

tracker = MultiPersonPoseTracker(
    max_distance=250.0,
    max_missing_frames=8,
    window_size=7,
    alpha=0.15,
)


# ============================================================
# LOAD SAVED PROXIMITY CONFIGURATION
# ============================================================

print(
    "Loading proximity configuration..."
)

proximity_manager = MultiPersonProximity(
    config_path=CONFIG_PATH
)

print(
    "Proximity configuration loaded."
)

print(
    f"Reference distance: "
    f"{proximity_manager.reference_distance_cm:.1f} cm"
)

print(
    f"Reference face width: "
    f"{proximity_manager.reference_face_width_px:.1f} px"
)

print()


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    raise RuntimeError(
        "Could not open webcam."
    )


print("=" * 65)
print("SafeSight - AUTOMATIC Multi Person Proximity")
print("=" * 65)
print()
print("No manual calibration is required.")
print()
print("Q = quit")
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

    # --------------------------------------------------------
    # 1. DETECT
    # --------------------------------------------------------

    faces = detector.detect(
        frame
    )

    # --------------------------------------------------------
    # 2. TRACK
    # --------------------------------------------------------

    tracks = tracker.update(
        faces,
        frame.shape
    )

    # --------------------------------------------------------
    # 3. ACTIVE IDS
    # --------------------------------------------------------

    active_ids = [
        track["track_id"]
        for track in tracks
    ]

    proximity_manager.cleanup(
        active_ids
    )

    # --------------------------------------------------------
    # 4. PROCESS EACH PERSON
    # --------------------------------------------------------

    for track in tracks:

        track_id = track[
            "track_id"
        ]

        face = track[
            "face"
        ]

        x, y, w, h = face[
            "bbox"
        ]

        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)

        # ----------------------------------------------------
        # PROXIMITY
        # ----------------------------------------------------

        result = proximity_manager.update(
            track_id,
            w
        )

        raw_distance = result[
            "raw_distance_cm"
        ]

        stable_distance = result[
            "distance_cm"
        ]

        proximity = result[
            "proximity"
        ]

        # ----------------------------------------------------
        # COLOR
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

        # ----------------------------------------------------
        # FACE BOX
        # ----------------------------------------------------

        cv2.rectangle(
            display,
            (x, y),
            (x + w, y + h),
            box_color,
            2,
        )

        # ----------------------------------------------------
        # PERSON
        # ----------------------------------------------------

        cv2.putText(
            display,
            f"Person {track_id}",
            (x, y - 55),
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
            (x, y - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            box_color,
            2,
        )

        # ----------------------------------------------------
        # STABLE DISTANCE
        # ----------------------------------------------------

        if stable_distance is not None:

            cv2.putText(
                display,
                f"Distance: {stable_distance:.1f} cm",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

        # ----------------------------------------------------
        # RAW DISTANCE
        # ----------------------------------------------------

        if raw_distance is not None:

            cv2.putText(
                display,
                f"Raw: {raw_distance:.1f} cm",
                (x, y + h + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )

        # ----------------------------------------------------
        # FACE WIDTH
        # ----------------------------------------------------

        cv2.putText(
            display,
            f"Face width: {w}px",
            (x, y + h + 45),
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

    cv2.putText(
        display,
        f"Active tracks: {len(tracks)}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )

    cv2.putText(
        display,
        "AUTO PROXIMITY - NO CALIBRATION",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
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

    if key == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print(
    "Automatic proximity test stopped."
)