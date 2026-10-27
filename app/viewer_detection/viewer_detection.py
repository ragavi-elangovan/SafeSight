import cv2
import numpy as np

from insightface.app import FaceAnalysis
from insightface.utils import face_align

from app.viewer_detection.yunet_detector import YuNetDetector
from app.viewer_detection.matcher import find_best_match
from app.viewer_detection.pose_tracker import MultiPersonPoseTracker
from app.viewer_detection.head_pose import (
    estimate_head_pose,
    classify_head_direction,
    PoseFilter,
)
from app.viewer_detection.gaze_estimator import GazeEstimator
from app.viewer_detection.multi_person_proximity import MultiPersonProximity


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_PATH = "data/embeddings/RAGAVI.npy"
PROXIMITY_CONFIG = "data/config/proximity_config.json"

MATCH_THRESHOLD = 0.45

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720


# ============================================================
# LOAD ENROLLED EMBEDDINGS
# ============================================================

print("Loading enrolled embeddings...")

enrolled_embeddings = np.load(EMBEDDING_PATH)

print(f"Loaded embeddings: {enrolled_embeddings.shape}")


# ============================================================
# LOAD YUNET
# ============================================================

print("Loading YuNet...")

detector = YuNetDetector(
    model_path="models/face_detection_yunet_2026may.onnx"
)

print("YuNet loaded.")


# ============================================================
# LOAD ARCFACE
# ============================================================

print("Loading ArcFace...")

face_app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

face_app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)

recognition_model = face_app.models["recognition"]

print("ArcFace loaded.")


# ============================================================
# LOAD TRACKER
# ============================================================

tracker = MultiPersonPoseTracker(
    max_distance=250.0,
    max_missing_frames=8,
    window_size=7,
    alpha=0.15
)


# ============================================================
# LOAD GAZE ESTIMATOR
# ============================================================

print("Loading gaze estimator...")

gaze_estimator = GazeEstimator()

print("Gaze estimator loaded.")


# ============================================================
# LOAD PROXIMITY MANAGER
# ============================================================

print("Loading proximity configuration...")

proximity_manager = MultiPersonProximity(
    config_path=PROXIMITY_CONFIG
)

print(
    f"Reference distance: "
    f"{proximity_manager.reference_distance_cm:.1f} cm"
)

print(
    f"Reference face width: "
    f"{proximity_manager.reference_face_width_px:.1f} px"
)


# ============================================================
# HEAD POSE FILTERS
# ============================================================

pose_filters = {}


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")


cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)


print()
print("=" * 60)
print("SafeSight - UNIFIED VIEWER DETECTION")
print("=" * 60)
print("Q = Quit")
print("=" * 60)


# ============================================================
# MAIN LOOP
# ============================================================

try:

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Failed to read camera frame.")
            break


        # ----------------------------------------------------
        # 1. FACE DETECTION
        # ----------------------------------------------------

        faces = detector.detect(frame)


        # ----------------------------------------------------
        # 2. MULTI-PERSON TRACKING
        # ----------------------------------------------------

        tracks = tracker.update(
            faces,
            frame.shape
        )


        active_track_ids = [
            track["track_id"]
            for track in tracks
        ]


        # Remove proximity data for disappeared people
        proximity_manager.cleanup(
            active_track_ids
        )


        # Remove pose filters for disappeared people
        active_ids_set = set(active_track_ids)

        for track_id in list(pose_filters.keys()):

            if track_id not in active_ids_set:
                del pose_filters[track_id]


        # ----------------------------------------------------
        # 3. PROCESS EACH PERSON
        # ----------------------------------------------------

        for track in tracks:

            track_id = track["track_id"]

            face = track["face"]

            x, y, w, h = face["bbox"]


            # =================================================
            # IDENTITY RECOGNITION
            # =================================================

            landmark_array = np.array(
                [
                    face["landmarks"]["right_eye"],
                    face["landmarks"]["left_eye"],
                    face["landmarks"]["nose"],
                    face["landmarks"]["right_mouth"],
                    face["landmarks"]["left_mouth"],
                ],
                dtype=np.float32
            )


            # ArcFace alignment
            aligned_face = face_align.norm_crop(
                frame,
                landmark=landmark_array,
                image_size=112
            )


            # Generate embedding
            embedding = (
                recognition_model
                .get_feat(aligned_face)
                .flatten()
            )


            # Normalize
            norm = np.linalg.norm(embedding)

            if norm > 0:
                embedding = embedding / norm


            # Match against enrolled person
            match_result = find_best_match(
                embedding,
                enrolled_embeddings,
                threshold=MATCH_THRESHOLD
            )


            score = match_result["score"]


            if match_result["matched"]:

                identity = "RAGAVI"

            else:

                identity = "UNKNOWN"


            # =================================================
            # HEAD POSE
            # =================================================

            if track_id not in pose_filters:

                pose_filters[track_id] = PoseFilter(
                    window_size=7,
                    alpha=0.15
                )


            raw_pose = estimate_head_pose(
                face,
                frame.shape
            )


            filtered_pose = pose_filters[
                track_id
            ].update(raw_pose)


            head_direction = classify_head_direction(
                filtered_pose
            )


            yaw = filtered_pose["yaw"]
            pitch = filtered_pose["pitch"]
            roll = filtered_pose["roll"]


            # =================================================
            # GAZE
            # =================================================

            gaze_result = gaze_estimator.estimate(
                frame,
                face
            )


            if gaze_result is None:

                gaze_direction = "UNKNOWN"

            elif isinstance(gaze_result, dict):

                gaze_direction = gaze_result.get(
                    "direction",
                    "UNKNOWN"
                )

            else:

                gaze_direction = str(
                    gaze_result
                )


            # =================================================
            # PROXIMITY
            # =================================================

            proximity_result = proximity_manager.update(
                track_id,
                w
            )


            distance_cm = proximity_result[
                "distance_cm"
            ]

            raw_distance_cm = proximity_result[
                "raw_distance_cm"
            ]

            proximity = proximity_result[
                "proximity"
            ]


            # =================================================
            # DRAW FACE BOX
            # =================================================

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )


            # =================================================
            # LABEL POSITION
            # =================================================

            label_x = x

            label_y = max(
                25,
                y - 10
            )


            # =================================================
            # PERSON ID
            # =================================================

            cv2.putText(
                frame,
                f"Person {track_id}",
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )


            # =================================================
            # IDENTITY
            # =================================================

            cv2.putText(
                frame,
                f"Identity: {identity}",
                (label_x, y + h + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )


            cv2.putText(
                frame,
                f"Match: {score:.2f}",
                (label_x, y + h + 47),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )


            # =================================================
            # HEAD
            # =================================================

            cv2.putText(
                frame,
                f"Head: {head_direction}",
                (label_x, y + h + 69),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )


            cv2.putText(
                frame,
                f"Yaw: {yaw:.1f}  Pitch: {pitch:.1f}",
                (label_x, y + h + 91),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )


            # =================================================
            # GAZE
            # =================================================

            cv2.putText(
                frame,
                f"Gaze: {gaze_direction}",
                (label_x, y + h + 113),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )


            # =================================================
            # PROXIMITY
            # =================================================

            cv2.putText(
                frame,
                f"Distance: {distance_cm:.1f} cm",
                (label_x, y + h + 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )


            cv2.putText(
                frame,
                f"Proximity: {proximity}",
                (label_x, y + h + 157),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )


            # =================================================
            # RAW DISTANCE
            # =================================================

            cv2.putText(
                frame,
                f"Raw: {raw_distance_cm:.1f} cm",
                (label_x, y + h + 179),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (180, 180, 180),
                1
            )


        # ====================================================
        # GLOBAL INFORMATION
        # ====================================================

        cv2.putText(
            frame,
            f"Faces detected: {len(faces)}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Active tracks: {len(tracks)}",
            (20, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )


        cv2.putText(
            frame,
            "SAFE SIGHT - VIEWER DETECTION",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


        # ----------------------------------------------------
        # SHOW FRAME
        # ----------------------------------------------------

        cv2.imshow(
            "SafeSight - Viewer Detection",
            frame
        )


        # ----------------------------------------------------
        # QUIT
        # ----------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break


finally:

    cap.release()

    cv2.destroyAllWindows()

    gaze_estimator.close()

    print()
    print("SafeSight Viewer Detection stopped.")