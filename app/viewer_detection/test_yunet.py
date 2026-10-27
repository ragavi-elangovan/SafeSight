import cv2

from app.viewer_detection.yunet_detector import YuNetDetector


MODEL_PATH = "models/face_detection_yunet_2026may.onnx"


def main():

    detector = YuNetDetector(
        model_path=MODEL_PATH
    )

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("YuNet face detection started.")
    print("Press Q to quit.")

    while True:

        ret, frame = camera.read()

        if not ret:
            print("Could not read camera frame.")
            break

        faces = detector.detect(frame)

        for face in faces:

            x, y, w, h = face["bbox"]
            confidence = face["confidence"]

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"Face: {confidence:.2f}",
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        cv2.putText(
            frame,
            f"Faces detected: {len(faces)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.imshow(
            "SafeSight - YuNet Face Detection",
            frame,
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()