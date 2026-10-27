import cv2
import numpy as np

from insightface.app import FaceAnalysis
from insightface.utils import face_align

from app.viewer_detection.yunet_detector import YuNetDetector
from app.viewer_detection.matcher import find_best_match


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/face_detection_yunet_2026may.onnx"

EMBEDDING_PATH = "data/embeddings/RAGAVI.npy"

PERSON_NAME = "RAGAVI"

# Temporary threshold.
# We will calibrate this properly later.
RECOGNITION_THRESHOLD = 0.45


# ============================================================
# ARCface EMBEDDING FROM ONE YUNET FACE
# ============================================================

def get_arcface_embedding(
    frame,
    face_info,
    recognition_model,
):
    """
    Generate an ArcFace embedding for ONE specific
    YuNet detection.

    YuNet gives:
        bbox
        5 landmarks

    We use those landmarks to align exactly that
    face to the ArcFace 112x112 input.
    """

    # --------------------------------------------------------
    # Get YuNet landmarks
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Validate landmarks
    # --------------------------------------------------------

    if landmark_array.shape != (5, 2):
        return None

    if not np.isfinite(landmark_array).all():
        return None

    # --------------------------------------------------------
    # ArcFace alignment
    #
    # ArcFace expects a 112 x 112 aligned face.
    # --------------------------------------------------------

    aligned_face = face_align.norm_crop(
        frame,
        landmark_array,
        image_size=112,
    )

    if aligned_face is None:
        return None

    # --------------------------------------------------------
    # Run ONLY the recognition model
    #
    # IMPORTANT:
    # We are NOT calling face_app.get(frame).
    # --------------------------------------------------------

    embedding = recognition_model.get_feat(
        aligned_face
    ).flatten()

    # --------------------------------------------------------
    # Convert to float32
    # --------------------------------------------------------

    embedding = embedding.astype(
        np.float32
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    norm = np.linalg.norm(
        embedding
    )

    if norm == 0:
        return None

    embedding = embedding / norm

    return embedding


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Load enrolled embeddings
    # --------------------------------------------------------

    print("Loading enrolled embeddings...")

    enrolled_embeddings = np.load(
        EMBEDDING_PATH
    ).astype(np.float32)

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

    # --------------------------------------------------------
    # 2. Load YuNet
    # --------------------------------------------------------

    print("Loading YuNet...")

    detector = YuNetDetector(
        model_path=MODEL_PATH
    )

    print("YuNet ready.")

    # --------------------------------------------------------
    # 3. Load InsightFace
    #
    # We only need its recognition model.
    # Its detector will NOT be used.
    # --------------------------------------------------------

    print("Loading ArcFace recognition model...")

    face_app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"],
    )

    face_app.prepare(
        ctx_id=0,
        det_size=(640, 640),
    )

    recognition_model = face_app.models[
        "recognition"
    ]

    print("ArcFace recognition model ready.")

    # --------------------------------------------------------
    # 4. Open camera
    # --------------------------------------------------------

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError(
            "Could not open webcam."
        )

    print()
    print("========================================")
    print("       SAFESIGHT RECOGNITION TEST")
    print("========================================")
    print()
    print(f"Enrolled person: {PERSON_NAME}")
    print()
    print("Press Q to quit.")
    print()

    # --------------------------------------------------------
    # 5. Recognition loop
    # --------------------------------------------------------

    while True:

        ret, frame = camera.read()

        if not ret:
            print(
                "Could not read camera frame."
            )
            break

        # ----------------------------------------------------
        # Detect ALL faces using YuNet
        # ----------------------------------------------------

        faces = detector.detect(frame)

        # ----------------------------------------------------
        # Process EACH YuNet face separately
        # ----------------------------------------------------

        for face_info in faces:

            x, y, w, h = face_info["bbox"]

            detection_confidence = (
                face_info["confidence"]
            )

            # ------------------------------------------------
            # Generate embedding ONLY for this face
            # ------------------------------------------------

            embedding = get_arcface_embedding(
                frame,
                face_info,
                recognition_model,
            )

            if embedding is None:

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 165, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    "NO EMBEDDING",
                    (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 165, 255),
                    2,
                )

                continue

            # ------------------------------------------------
            # Compare THIS face against RAGAVI embeddings
            # ------------------------------------------------

            result = find_best_match(
                embedding,
                enrolled_embeddings,
                threshold=RECOGNITION_THRESHOLD,
            )

            score = result["score"]

            # ------------------------------------------------
            # Recognition result
            # ------------------------------------------------

            if result["matched"]:

                label = (
                    f"{PERSON_NAME} "
                    f"{score:.2f}"
                )

                box_color = (
                    0,
                    255,
                    0,
                )

            else:

                label = (
                    f"UNKNOWN "
                    f"{score:.2f}"
                )

                box_color = (
                    0,
                    0,
                    255,
                )

            # ------------------------------------------------
            # Draw bounding box
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                box_color,
                2,
            )

            # ------------------------------------------------
            # Draw recognition label
            # ------------------------------------------------

            cv2.putText(
                frame,
                label,
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                box_color,
                2,
            )

            # ------------------------------------------------
            # Draw YuNet confidence
            # ------------------------------------------------

            cv2.putText(
                frame,
                f"Detection: {detection_confidence:.2f}",
                (x, y + h + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                box_color,
                1,
            )

        # ----------------------------------------------------
        # Display number of faces
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"Faces detected: {len(faces)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        # ----------------------------------------------------
        # Display camera
        # ----------------------------------------------------

        cv2.imshow(
            "SafeSight - Recognition",
            frame,
        )

        # ----------------------------------------------------
        # Quit
        # ----------------------------------------------------

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    camera.release()
    cv2.destroyAllWindows()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()