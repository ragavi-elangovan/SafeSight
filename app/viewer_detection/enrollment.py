import cv2
import numpy as np
from pathlib import Path

from insightface.app import FaceAnalysis
from insightface.utils import face_align

from app.viewer_detection.yunet_detector import YuNetDetector


MODEL_PATH = "models/face_detection_yunet_2026may.onnx"
EMBEDDING_DIR = Path("data/embeddings")

NUM_SAMPLES = 10


def get_arcface_embedding(frame, face_info, recognition_model):
    """
    Convert one YuNet-detected face into a normalized
    512-dimensional ArcFace embedding.
    """

    landmarks = face_info["landmarks"]

    landmark_array = np.array(
        [
            landmarks["right_eye"],
            landmarks["left_eye"],
            landmarks["nose"],
            landmarks["right_mouth"],
            landmarks["left_mouth"],
        ],
        dtype=np.float32,
    )

    # Validate landmarks
    if landmark_array.shape != (5, 2):
        return None

    if not np.isfinite(landmark_array).all():
        return None

    # Align face using the same ArcFace preprocessing
    aligned_face = face_align.norm_crop(
        frame,
        landmark_array,
        image_size=112,
    )

    if aligned_face is None:
        return None

    # Generate ArcFace embedding
    embedding = recognition_model.get_feat(aligned_face).flatten()

    embedding = embedding.astype(np.float32)

    # Normalize embedding
    norm = np.linalg.norm(embedding)

    if norm == 0:
        return None

    embedding = embedding / norm

    return embedding


def main():

    print()
    print("========================================")
    print("       SAFESIGHT FACE ENROLLMENT")
    print("========================================")
    print()

    # -----------------------------------------
    # Get person's name
    # -----------------------------------------

    person_name = input("Enter person's name: ").strip()

    if not person_name:
        print("Name cannot be empty.")
        return

    safe_name = "".join(
        c for c in person_name
        if c.isalnum() or c in ("_", "-")
    )

    if not safe_name:
        print("Invalid name.")
        return

    # -----------------------------------------
    # Create embedding directory
    # -----------------------------------------

    EMBEDDING_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    embedding_path = EMBEDDING_DIR / f"{safe_name}.npy"

    # -----------------------------------------
    # Load YuNet
    # -----------------------------------------

    print()
    print("Loading YuNet...")

    detector = YuNetDetector(
        model_path=MODEL_PATH
    )

    print("YuNet ready.")

    # -----------------------------------------
    # Load ArcFace
    # -----------------------------------------

    print()
    print("Loading ArcFace recognition model...")

    face_app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )

    face_app.prepare(
        ctx_id=0,
        det_size=(640, 640)
    )

    recognition_model = face_app.models["recognition"]

    print("ArcFace recognition model ready.")

    # -----------------------------------------
    # Open camera
    # -----------------------------------------

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Could not open webcam.")

    embeddings = []

    print()
    print("========================================")
    print("           ENROLLMENT STARTED")
    print("========================================")
    print()
    print(f"Person : {person_name}")
    print(f"Samples: {NUM_SAMPLES}")
    print()
    print("Instructions:")
    print("- Keep only ONE face visible.")
    print("- Look toward the camera.")
    print("- Move your face slightly between samples.")
    print("- Press Q to cancel.")
    print()

    while len(embeddings) < NUM_SAMPLES:

        ret, frame = camera.read()

        if not ret:
            print("Could not read camera frame.")
            break

        # -----------------------------------------
        # Detect faces using YuNet
        # -----------------------------------------

        faces = detector.detect(frame)

        # Draw number of detected faces
        cv2.putText(
            frame,
            f"Faces detected: {len(faces)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        # -----------------------------------------
        # We need exactly one face
        # -----------------------------------------

        if len(faces) == 1:

            face_info = faces[0]

            x, y, w, h = face_info["bbox"]

            # Draw face box
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2,
            )

            # -----------------------------------------
            # Generate embedding
            # -----------------------------------------

            embedding = get_arcface_embedding(
                frame,
                face_info,
                recognition_model,
            )

            if embedding is not None:

                # Only capture when SPACE is pressed
                cv2.putText(
                    frame,
                    "Press SPACE to capture",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Samples: {len(embeddings)}/{NUM_SAMPLES}",
                    (20, 115),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow(
                    "SafeSight - Enrollment",
                    frame
                )

                key = cv2.waitKey(1) & 0xFF

                # SPACE = capture sample
                if key == 32:

                    embeddings.append(embedding)

                    print(
                        f"Captured sample "
                        f"{len(embeddings)}/{NUM_SAMPLES}"
                    )

                    continue

                # Q = quit
                if key == ord("q"):
                    print("Enrollment cancelled.")
                    break

                continue

            else:

                cv2.putText(
                    frame,
                    "Could not generate embedding",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

        elif len(faces) == 0:

            cv2.putText(
                frame,
                "No face detected",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        else:

            cv2.putText(
                frame,
                "Only ONE face should be visible",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        cv2.imshow(
            "SafeSight - Enrollment",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("Enrollment cancelled.")
            break

    # -----------------------------------------
    # Release camera
    # -----------------------------------------

    camera.release()
    cv2.destroyAllWindows()

    # -----------------------------------------
    # Save embeddings
    # -----------------------------------------

    if len(embeddings) != NUM_SAMPLES:

        print()
        print(
            f"Enrollment not completed. "
            f"Only {len(embeddings)} samples captured."
        )

        return

    embeddings_array = np.stack(
        embeddings
    ).astype(np.float32)

    np.save(
        embedding_path,
        embeddings_array
    )

    print()
    print("========================================")
    print("       ENROLLMENT COMPLETED")
    print("========================================")
    print()
    print(f"Person       : {person_name}")
    print(f"Samples      : {len(embeddings)}")
    print(f"Shape        : {embeddings_array.shape}")
    print(f"Saved to     : {embedding_path}")
    print()
    print("Raw camera images were NOT saved.")
    print("Only face embeddings were stored.")
    print()

if __name__ == "__main__":
    main()