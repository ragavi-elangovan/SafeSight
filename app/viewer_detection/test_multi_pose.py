import cv2

from app.viewer_detection.yunet_detector import (
    YuNetDetector,
)

from app.viewer_detection.pose_tracker import (
    MultiPersonPoseTracker,
)

from app.viewer_detection.head_pose import (
    classify_head_direction,
)


MODEL_PATH = (
    "models/face_detection_yunet_2026may.onnx"
)


# =========================================================
# TRACKER SETTINGS
# =========================================================

MAX_DISTANCE = 250.0

MAX_MISSING_FRAMES = 8

POSE_WINDOW_SIZE = 7

POSE_ALPHA = 0.15


def main():

    print()
    print("========================================")
    print("     SAFESIGHT MULTI-PERSON POSE TEST")
    print("========================================")
    print()
    print("Press Q to quit.")
    print()

    # =====================================================
    # LOAD YUNET
    # =====================================================

    print("Loading YuNet...")

    detector = YuNetDetector(
        model_path=MODEL_PATH
    )

    print("YuNet ready.")

    # =====================================================
    # CREATE TRACKER
    # =====================================================

    tracker = MultiPersonPoseTracker(
        max_distance=MAX_DISTANCE,
        max_missing_frames=MAX_MISSING_FRAMES,
        window_size=POSE_WINDOW_SIZE,
        alpha=POSE_ALPHA,
    )

    # =====================================================
    # OPEN CAMERA
    # =====================================================

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        raise RuntimeError(
            "Could not open webcam."
        )

    # =====================================================
    # CAMERA LOOP
    # =====================================================

    while True:

        ret, frame = camera.read()

        if not ret:

            print(
                "Could not read camera frame."
            )

            break

        # =================================================
        # FACE DETECTION
        # =================================================

        faces = detector.detect(
            frame
        )

        # =================================================
        # UPDATE TRACKER
        # =================================================

        tracked_faces = (
            tracker.update(
                faces,
                frame.shape,
            )
        )

        # =================================================
        # DRAW TRACKED PEOPLE
        # =================================================

        for item in tracked_faces:

            track_id = item[
                "track_id"
            ]

            face_info = item[
                "face"
            ]

            pose = item[
                "pose"
            ]

            x, y, w, h = (
                face_info["bbox"]
            )

            # -------------------------------------------------
            # Direction
            # -------------------------------------------------

            direction = (
                classify_head_direction(
                    pose
                )
            )

            # -------------------------------------------------
            # Bounding box
            # -------------------------------------------------

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2,
            )

            # -------------------------------------------------
            # Person ID
            # -------------------------------------------------

            cv2.putText(
                frame,
                f"Person {track_id}",
                (
                    x,
                    max(
                        y - 12,
                        25,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2,
            )

            # -------------------------------------------------
            # Pose information
            # -------------------------------------------------

            if pose is not None:

                cv2.putText(
                    frame,
                    f"Yaw: {pose['yaw']:.1f}",
                    (
                        x,
                        y + h + 22,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1,
                )

                cv2.putText(
                    frame,
                    f"Pitch: {pose['pitch']:.1f}",
                    (
                        x,
                        y + h + 42,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1,
                )

                cv2.putText(
                    frame,
                    f"Roll: {pose['roll']:.1f}",
                    (
                        x,
                        y + h + 62,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1,
                )

                # -------------------------------------------------
                # Direction
                # -------------------------------------------------

                cv2.putText(
                    frame,
                    direction,
                    (
                        x,
                        y + h + 86,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                )

            # -------------------------------------------------
            # Facial landmarks
            # -------------------------------------------------

            for point in (
                face_info[
                    "landmarks"
                ].values()
            ):

                px, py = point

                cv2.circle(
                    frame,
                    (
                        int(px),
                        int(py),
                    ),
                    3,
                    (0, 0, 255),
                    -1,
                )

        # =================================================
        # GLOBAL STATUS
        # =================================================

        cv2.putText(
            frame,
            f"Faces detected: {len(faces)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            f"Active tracks: {len(tracker.tracks)}",
            (20, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # =================================================
        # DISPLAY
        # =================================================

        cv2.imshow(
            "SafeSight - Multi Person Pose",
            frame,
        )

        # =================================================
        # KEYBOARD
        # =================================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    # =====================================================
    # CLEANUP
    # =====================================================

    camera.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()