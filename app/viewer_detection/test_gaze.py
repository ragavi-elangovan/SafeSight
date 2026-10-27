import cv2

from app.viewer_detection.yunet_detector import (
    YuNetDetector,
)

from app.viewer_detection.gaze_estimator import (
    GazeEstimator,
)


MODEL_PATH = (
    "models/face_detection_yunet_2026may.onnx"
)


def main():

    print()
    print("========================================")
    print("       SAFESIGHT GAZE TEST")
    print("========================================")
    print()
    print("Look at the camera, then move only your eyes.")
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
    # LOAD GAZE ESTIMATOR
    # =====================================================

    print("Loading MediaPipe Face Mesh...")

    gaze_estimator = GazeEstimator()

    print("Gaze estimator ready.")

    # =====================================================
    # CAMERA
    # =====================================================

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError(
            "Could not open webcam."
        )

    # =====================================================
    # LOOP
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

        faces = detector.detect(
            frame
        )

        # -------------------------------------------------
        # Process each face
        # -------------------------------------------------

        for face_info in faces:

            x, y, w, h = (
                face_info["bbox"]
            )

            gaze = gaze_estimator.estimate(
                frame,
                face_info,
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
            # Gaze
            # -------------------------------------------------

            if gaze is not None:

                direction = gaze[
                    "direction"
                ]

                horizontal = gaze[
                    "horizontal"
                ]

                vertical = gaze[
                    "vertical"
                ]

                cv2.putText(
                    frame,
                    f"Gaze: {direction}",
                    (
                        x,
                        y + h + 25,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    f"H: {horizontal:.2f}",
                    (
                        x,
                        y + h + 48,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1,
                )

                cv2.putText(
                    frame,
                    f"V: {vertical:.2f}",
                    (
                        x,
                        y + h + 68,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1,
                )

            else:

                cv2.putText(
                    frame,
                    "Gaze unavailable",
                    (
                        x,
                        y + h + 25,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    2,
                )

            # -------------------------------------------------
            # YuNet landmarks
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

        # -------------------------------------------------
        # Face count
        # -------------------------------------------------

        cv2.putText(
            frame,
            f"Faces detected: {len(faces)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        # -------------------------------------------------
        # Display
        # -------------------------------------------------

        cv2.imshow(
            "SafeSight - Gaze",
            frame,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    # =====================================================
    # CLEANUP
    # =====================================================

    camera.release()

    gaze_estimator.close()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()