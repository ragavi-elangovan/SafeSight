import cv2
import numpy as np

from insightface.app import FaceAnalysis
from insightface.utils import face_align

from app.viewer_detection.yunet_detector import YuNetDetector
from app.viewer_detection.matcher import find_best_match
from app.viewer_detection.pose_tracker import MultiPersonPoseTracker
from app.viewer_detection.head_pose import classify_head_direction


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_PATH = (
    "models/face_detection_yunet_2026may.onnx"
)

EMBEDDING_PATH = (
    "data/embeddings/RAGAVI.npy"
)

PERSON_NAME = "RAGAVI"

RECOGNITION_THRESHOLD = 0.45


# =========================================================
# ARCFACE EMBEDDING
# =========================================================

def get_arcface_embedding(
    frame,
    face_info,
    recognition_model,
):
    """
    Generate a 512-dimensional ArcFace embedding
    for one specific YuNet-detected face.
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

    if landmark_array.shape != (5, 2):
        return None

    if not np.isfinite(
        landmark_array
    ).all():
        return None

    # -----------------------------------------------------
    # Align face
    # -----------------------------------------------------

    aligned_face = face_align.norm_crop(
        frame,
        landmark_array,
        image_size=112,
    )

    if aligned_face is None:
        return None

    # -----------------------------------------------------
    # Generate embedding
    # -----------------------------------------------------

    embedding = recognition_model.get_feat(
        aligned_face
    ).flatten()

    embedding = embedding.astype(
        np.float32
    )

    # -----------------------------------------------------
    # Normalize
    # -----------------------------------------------------

    norm = np.linalg.norm(
        embedding
    )

    if norm == 0:
        return None

    embedding = embedding / norm

    return embedding


# =========================================================
# MAIN
# =========================================================

def main():

    print()
    print("========================================")
    print("     SAFESIGHT IDENTITY + POSE TEST")
    print("========================================")
    print()
    print("Press Q to quit.")
    print()

    # =====================================================
    # LOAD ENROLLED EMBEDDINGS
    # =====================================================

    print(
        "Loading enrolled embeddings..."
    )

    enrolled_embeddings = np.load(
        EMBEDDING_PATH
    ).astype(
        np.float32
    )

    print(
        "Loaded embeddings:",
        enrolled_embeddings.shape,
    )

    if enrolled_embeddings.ndim != 2:
        raise ValueError(
            "Enrollment file must have shape (N, 512)."
        )

    if enrolled_embeddings.shape[1] != 512:
        raise ValueError(
            "Expected 512-dimensional embeddings."
        )

    # =====================================================
    # LOAD YUNET
    # =====================================================

    print()
    print("Loading YuNet...")

    detector = YuNetDetector(
        model_path=MODEL_PATH
    )

    print("YuNet ready.")

    # =====================================================
    # LOAD ARCFACE
    # =====================================================

    print()
    print(
        "Loading ArcFace recognition model..."
    )

    face_app = FaceAnalysis(
        name="buffalo_l",
        providers=[
            "CPUExecutionProvider"
        ],
    )

    face_app.prepare(
        ctx_id=0,
        det_size=(640, 640),
    )

    recognition_model = (
        face_app.models["recognition"]
    )

    print(
        "ArcFace recognition model ready."
    )

    # =====================================================
    # MULTI-PERSON POSE TRACKER
    # =====================================================

    tracker = MultiPersonPoseTracker(
        max_distance=250.0,
        max_missing_frames=8,
        window_size=7,
        alpha=0.15,
    )

    # =====================================================
    # OPEN CAMERA
    # =====================================================

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError(
            "Could not open webcam."
        )

    print()
    print("========================================")
    print("       IDENTITY + POSE STARTED")
    print("========================================")
    print()

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

        faces = detector.detect(
            frame
        )

        # -------------------------------------------------
        # Update pose tracker
        # -------------------------------------------------

        tracked_faces = tracker.update(
            faces,
            frame.shape,
        )

        # -------------------------------------------------
        # Process every person
        # -------------------------------------------------

        for item in tracked_faces:

            track_id = item["track_id"]

            face_info = item["face"]

            pose = item["pose"]

            # -------------------------------------------------
            # ArcFace embedding
            # -------------------------------------------------

            embedding = get_arcface_embedding(
                frame,
                face_info,
                recognition_model,
            )

            identity = "UNKNOWN"
            score = 0.0

            # -------------------------------------------------
            # Identity matching
            # -------------------------------------------------

            if embedding is not None:

                result = find_best_match(
                    embedding,
                    enrolled_embeddings,
                    threshold=RECOGNITION_THRESHOLD,
                )

                score = result["score"]

                if result["matched"]:
                    identity = PERSON_NAME

            # -------------------------------------------------
            # Bounding box
            # -------------------------------------------------

            x, y, w, h = face_info["bbox"]

            if identity == PERSON_NAME:
                box_color = (0, 255, 0)
            else:
                box_color = (0, 0, 255)

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                box_color,
                2,
            )

            # -------------------------------------------------
            # Identity label
            # -------------------------------------------------

            if identity == PERSON_NAME:

                identity_text = (
                    f"{identity} {score:.2f}"
                )

            else:

                identity_text = (
                    f"UNKNOWN {score:.2f}"
                )

            cv2.putText(
                frame,
                identity_text,
                (
                    x,
                    max(y - 10, 20),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                box_color,
                2,
            )

            # -------------------------------------------------
            # Person / Track ID
            # -------------------------------------------------

            cv2.putText(
                frame,
                f"Person {track_id}",
                (
                    x,
                    y + h + 22,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2,
            )

            # -------------------------------------------------
            # Head direction
            # -------------------------------------------------

            direction = classify_head_direction(
                pose
            )

            cv2.putText(
                frame,
                f"Head: {direction}",
                (
                    x,
                    y + h + 45,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )

            # -------------------------------------------------
            # Pose values
            # -------------------------------------------------

            if pose is not None:

                cv2.putText(
                    frame,
                    f"Yaw: {pose['yaw']:.1f}",
                    (
                        x,
                        y + h + 67,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 0),
                    1,
                )

                cv2.putText(
                    frame,
                    f"Pitch: {pose['pitch']:.1f}",
                    (
                        x,
                        y + h + 87,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 0),
                    1,
                )

            # -------------------------------------------------
            # Landmarks
            # -------------------------------------------------

            for point in face_info[
                "landmarks"
            ].values():

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

        # =====================================================
        # GLOBAL STATUS
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

        cv2.putText(
            frame,
            f"Active tracks: {len(tracker.tracks)}",
            (20, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # =====================================================
        # DISPLAY
        # =====================================================

        cv2.imshow(
            "SafeSight - Identity + Pose",
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