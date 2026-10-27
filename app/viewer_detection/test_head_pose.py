import cv2

from app.viewer_detection.yunet_detector import YuNetDetector

from app.viewer_detection.head_pose import (
    estimate_head_pose,
    draw_head_pose,
    PoseFilter,
)


MODEL_PATH = "models/face_detection_yunet_2026may.onnx"


# =========================================================
# FILTER SETTINGS
# =========================================================

# Number of frames used for median filtering.
#
# 5  -> faster
# 7  -> balanced
# 9  -> smoother
#
POSE_WINDOW_SIZE = 7


# EMA smoothing.
#
# Lower = smoother
# Higher = faster
#
POSE_ALPHA = 0.15


def main():

    print()
    print("========================================")
    print("       SAFESIGHT HEAD POSE TEST")
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
    # OPEN CAMERA
    # =====================================================

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError(
            "Could not open webcam."
        )

    # =====================================================
    # CREATE POSE FILTER
    # =====================================================

    pose_filter = PoseFilter(
        window_size=POSE_WINDOW_SIZE,
        alpha=POSE_ALPHA,
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

        # -------------------------------------------------
        # Detect faces
        # -------------------------------------------------

        faces = detector.detect(frame)

        # -------------------------------------------------
        # No faces
        # -------------------------------------------------

        if len(faces) == 0:

            pose_filter.reset()

        # -------------------------------------------------
        # Process faces
        # -------------------------------------------------

        for face_index, face_info in enumerate(faces):

            x, y, w, h = face_info["bbox"]

            # -------------------------------------------------
            # Estimate raw pose
            # -------------------------------------------------

            raw_pose = estimate_head_pose(
                face_info,
                frame.shape,
            )

            # -------------------------------------------------
            # Apply filtering
            # -------------------------------------------------
            #
            # This test is currently optimized for
            # a single person.
            #
            # Multi-person tracking will later maintain
            # one PoseFilter per person.
            # -------------------------------------------------

            if face_index == 0:

                pose = pose_filter.update(
                    raw_pose
                )

            else:

                pose = raw_pose

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
            # Draw landmarks
            # -------------------------------------------------

            landmarks = face_info["landmarks"]

            for point in landmarks.values():

                px, py = point

                cv2.circle(
                    frame,
                    (int(px), int(py)),
                    3,
                    (0, 0, 255),
                    -1,
                )

            # -------------------------------------------------
            # Draw pose
            # -------------------------------------------------

            draw_head_pose(
                frame,
                face_info,
                pose,
            )

        # =====================================================
        # FACE COUNT
        # =====================================================

        cv2.putText(
            frame,
            f"Faces detected: {len(faces)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        # =====================================================
        # DISPLAY
        # =====================================================

        cv2.imshow(
            "SafeSight - Head Pose",
            frame,
        )

        # =====================================================
        # KEYBOARD
        # =====================================================

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