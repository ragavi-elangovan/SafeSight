import cv2
import numpy as np

from insightface.app import FaceAnalysis
from app.viewer_detection.yunet_detector import YuNetDetector


MODEL_PATH = "models/face_detection_yunet_2026may.onnx"


def main():

    # -----------------------------
    # 1. Load YuNet
    # -----------------------------
    detector = YuNetDetector(
        model_path=MODEL_PATH
    )

    # -----------------------------
    # 2. Load ArcFace
    # -----------------------------
    print("Loading ArcFace...")

    face_app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"],
    )

    face_app.prepare(
        ctx_id=0,
        det_size=(640, 640),
    )

    print("ArcFace ready.")

    # -----------------------------
    # 3. Open camera
    # -----------------------------
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("Camera started.")
    print("Press E to capture embedding.")
    print("Press Q to quit.")

    while True:

        ret, frame = camera.read()

        if not ret:
            break

        # -----------------------------
        # 4. Detect faces with YuNet
        # -----------------------------
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
                f"Face {confidence:.2f}",
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        cv2.putText(
            frame,
            f"Faces: {len(faces)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.imshow(
            "SafeSight - ArcFace Embedding Test",
            frame,
        )

        key = cv2.waitKey(1) & 0xFF

        # -----------------------------
        # 5. Capture embedding
        # -----------------------------
        if key == ord("e"):

            if len(faces) != 1:

                print(
                    "Please make sure exactly ONE "
                    "face is visible."
                )

                continue

            print("Generating ArcFace embedding...")

            # InsightFace performs its own
            # alignment internally for this test.
            detected_faces = face_app.get(frame)

            if not detected_faces:

                print(
                    "ArcFace could not extract the face."
                )

                continue

            face = max(
                detected_faces,
                key=lambda f: (
                    f.bbox[2] - f.bbox[0]
                ) * (
                    f.bbox[3] - f.bbox[1]
                ),
            )

            embedding = face.embedding.astype(
                np.float32
            )

            # Normalize
            norm = np.linalg.norm(embedding)

            if norm == 0:

                print(
                    "Invalid embedding."
                )

                continue

            embedding = embedding / norm

            print()
            print("==============================")
            print("ArcFace embedding generated!")
            print(
                "Embedding shape:",
                embedding.shape,
            )
            print(
                "Embedding dimension:",
                len(embedding),
            )
            print(
                "Embedding norm:",
                np.linalg.norm(embedding),
            )
            print("==============================")
            print()

        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()