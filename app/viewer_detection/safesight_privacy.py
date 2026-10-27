import cv2
import time
import mss
import threading
import numpy as np

from copy import deepcopy

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
from app.viewer_detection.multi_person_proximity import (
    MultiPersonProximity,
)
from app.viewer_detection.sensitive_info_detector import (
    SensitiveInfoDetector,
)
from app.viewer_detection.screen_mask_overlay import (
    ScreenMaskOverlay,
)


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_PATH = "data/embeddings/RAGAVI.npy"

PROXIMITY_CONFIG = (
    "data/config/proximity_config.json"
)

MATCH_THRESHOLD = 0.45

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# OCR runs once per second in background
OCR_INTERVAL = 1.0

# AI camera processing:
# 1 = every frame
# 2 = every second frame
PROCESS_EVERY_N_FRAMES = 2


# ============================================================
# PRIVACY DECISION ENGINE
# ============================================================

class PrivacyDecisionEngine:

    def decide(
        self,
        identity,
        distance_cm,
        proximity,
        head_direction,
        gaze_direction,
    ):

        # ----------------------------------------------------
        # AUTHORIZED USER
        # ----------------------------------------------------

        if identity == "RAGAVI":

            return {
                "decision": "ALLOW",
                "risk": 0.0,
            }

        # ----------------------------------------------------
        # UNKNOWN - VERY CLOSE
        # ----------------------------------------------------

        if proximity == "VERY CLOSE":

            return {
                "decision": "BLOCK",
                "risk": 1.0,
            }

        # ----------------------------------------------------
        # UNKNOWN - CLOSE
        # ----------------------------------------------------

        if proximity == "CLOSE":

            return {
                "decision": "MASK",
                "risk": 0.8,
            }

        # ----------------------------------------------------
        # UNKNOWN - MEDIUM
        # ----------------------------------------------------

        if proximity == "MEDIUM":

            return {
                "decision": "MASK",
                "risk": 0.6,
            }

        # ----------------------------------------------------
        # UNKNOWN - FAR
        # ----------------------------------------------------

        if proximity == "FAR":

            return {
                "decision": "ALLOW",
                "risk": 0.15,
            }

        # ----------------------------------------------------
        # DEFAULT
        # ----------------------------------------------------

        return {
            "decision": "ALLOW",
            "risk": 0.0,
        }


# ============================================================
# BACKGROUND SCREEN OCR WORKER
# ============================================================

class ScreenOCRWorker:

    def __init__(
        self,
        monitor,
        interval=1.0,
    ):

        self.monitor = monitor

        self.interval = interval

        self.detector = (
            SensitiveInfoDetector()
        )

        self.running = False

        self.masking_requested = False

        self.lock = threading.Lock()

        self.latest_masks = []

        self.latest_detections = []

        self.thread = None

    # ========================================================
    # START
    # ========================================================

    def start(self):

        self.running = True

        self.thread = threading.Thread(
            target=self._worker,
            daemon=True,
        )

        self.thread.start()

    # ========================================================
    # ENABLE / DISABLE MASKING
    # ========================================================

    def set_masking(
        self,
        enabled,
    ):

        with self.lock:

            self.masking_requested = (
                enabled
            )

            # Immediately remove old
            # detections when masking
            # is disabled.

            if not enabled:

                self.latest_masks = []

                self.latest_detections = []

    # ========================================================
    # GET MASKS
    # ========================================================

    def get_masks(self):

        with self.lock:

            return deepcopy(
                self.latest_masks
            )

    # ========================================================
    # GET DETECTIONS
    # ========================================================

    def get_detections(self):

        with self.lock:

            return deepcopy(
                self.latest_detections
            )

    # ========================================================
    # BACKGROUND WORKER
    # ========================================================

    def _worker(self):

        print(
            "[OCR] Background OCR thread started."
        )

        try:

            with mss.MSS() as sct:

                while self.running:

                    # ----------------------------------------
                    # Check masking state
                    # ----------------------------------------

                    with self.lock:

                        should_mask = (
                            self.masking_requested
                        )

                    if not should_mask:

                        time.sleep(
                            0.15
                        )

                        continue

                    # ----------------------------------------
                    # Capture desktop
                    # ----------------------------------------

                    screenshot = np.array(
                        sct.grab(
                            self.monitor
                        )
                    )

                    screen_frame = (
                        screenshot[:, :, :3]
                    )

                    # ----------------------------------------
                    # Run Tesseract OCR
                    # ----------------------------------------

                    detections = (
                        self.detector.detect(
                            screen_frame
                        )
                    )

                    masks = []

                    for item in detections:

                        bbox = item.get(
                            "bbox"
                        )

                        if bbox is None:

                            continue

                        x, y, w, h = bbox

                        masks.append(
                            {
                                "bbox": (
                                    int(x),
                                    int(y),
                                    int(w),
                                    int(h),
                                )
                            }
                        )

                    # ----------------------------------------
                    # Store latest result
                    # ----------------------------------------

                    with self.lock:

                        self.latest_masks = masks

                        self.latest_detections = (
                            detections
                        )

                    # ----------------------------------------
                    # Wait
                    # ----------------------------------------

                    time.sleep(
                        self.interval
                    )

        except Exception as error:

            print(
                f"[OCR ERROR] {error}"
            )

        print(
            "[OCR] Background OCR thread stopped."
        )

    # ========================================================
    # STOP
    # ========================================================

    def stop(self):

        self.running = False

        if self.thread is not None:

            self.thread.join(
                timeout=2.0
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("SafeSight - AI PRIVACY PROTECTION SYSTEM")
    print("=" * 60)

    # ========================================================
    # LOAD EMBEDDINGS
    # ========================================================

    print(
        "Loading enrolled embeddings..."
    )

    enrolled_embeddings = np.load(
        EMBEDDING_PATH
    )

    print(
        f"Loaded embeddings: "
        f"{enrolled_embeddings.shape}"
    )

    # ========================================================
    # LOAD YUNET
    # ========================================================

    print(
        "Loading YuNet..."
    )

    detector = YuNetDetector(
        model_path=(
            "models/"
            "face_detection_yunet_2026may.onnx"
        )
    )

    print(
        "YuNet loaded."
    )

    # ========================================================
    # LOAD ARCFACE
    # ========================================================

    print(
        "Loading ArcFace..."
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
        "ArcFace loaded."
    )

    # ========================================================
    # TRACKER
    # ========================================================

    tracker = MultiPersonPoseTracker(
        max_distance=250.0,
        max_missing_frames=8,
        window_size=7,
        alpha=0.15,
    )

    # ========================================================
    # GAZE
    # ========================================================

    print(
        "Loading gaze estimator..."
    )

    gaze_estimator = GazeEstimator()

    print(
        "Gaze estimator loaded."
    )

    # ========================================================
    # PROXIMITY
    # ========================================================

    print(
        "Loading proximity configuration..."
    )

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

    # ========================================================
    # PRIVACY ENGINE
    # ========================================================

    privacy_engine = (
        PrivacyDecisionEngine()
    )

    # ========================================================
    # HEAD POSE FILTERS
    # ========================================================

    pose_filters = {}

    # ========================================================
    # CAMERA
    # ========================================================

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        raise RuntimeError(
            "Could not open webcam."
        )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        FRAME_WIDTH,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        FRAME_HEIGHT,
    )

    # ========================================================
    # SCREEN INFORMATION
    # ========================================================

    with mss.MSS() as sct:

        monitor = sct.monitors[1]

    screen_width = monitor["width"]
    screen_height = monitor["height"]

    screen_x = monitor["left"]
    screen_y = monitor["top"]

    print()
    print(
        f"Screen: "
        f"{screen_width}x{screen_height}"
    )

    # ========================================================
    # NATIVE OVERLAY
    # ========================================================

    overlay = ScreenMaskOverlay(

        screen_width=screen_width,

        screen_height=screen_height,

        screen_x=screen_x,

        screen_y=screen_y,
    )

    overlay.show()

    # ========================================================
    # BACKGROUND OCR
    # ========================================================

    ocr_worker = ScreenOCRWorker(

        monitor=monitor,

        interval=OCR_INTERVAL,
    )

    ocr_worker.start()

    # ========================================================
    # STATE
    # ========================================================

    masking_active = False

    last_overlay_masks = []

    frame_counter = 0

    # Last decisions are retained so
    # skipped camera frames do not
    # produce blank information.

    last_decisions = []

    # ========================================================
    # START
    # ========================================================

    print()
    print("=" * 60)
    print("SafeSight Privacy Protection ACTIVE")
    print("=" * 60)
    print()
    print("Q = Quit")
    print()
    print(
        "OCR runs in background."
    )
    print(
        "Camera AI processing runs every "
        f"{PROCESS_EVERY_N_FRAMES} frames."
    )
    print()

    # ========================================================
    # MAIN LOOP
    # ========================================================

    try:

        while True:

            # ------------------------------------------------
            # READ CAMERA
            # ------------------------------------------------

            ret, frame = cap.read()

            if not ret:

                print(
                    "Failed to read camera frame."
                )

                break

            frame_counter += 1

            # ------------------------------------------------
            # PROCESS AI ONLY EVERY NTH FRAME
            # ------------------------------------------------

            process_this_frame = (
                frame_counter
                % PROCESS_EVERY_N_FRAMES
                == 0
            )

            if process_this_frame:

                # ============================================
                # FACE DETECTION
                # ============================================

                faces = detector.detect(
                    frame
                )

                # ============================================
                # TRACKING
                # ============================================

                tracks = tracker.update(
                    faces,
                    frame.shape
                )

                active_track_ids = [

                    track["track_id"]

                    for track in tracks
                ]

                # ============================================
                # CLEAN PROXIMITY
                # ============================================

                proximity_manager.cleanup(
                    active_track_ids
                )

                # ============================================
                # CLEAN POSE FILTERS
                # ============================================

                active_ids_set = set(
                    active_track_ids
                )

                for track_id in list(
                    pose_filters.keys()
                ):

                    if (
                        track_id
                        not in active_ids_set
                    ):

                        del pose_filters[
                            track_id
                        ]

                # ============================================
                # CURRENT DECISIONS
                # ============================================

                current_decisions = []

                # ============================================
                # PROCESS EACH PERSON
                # ============================================

                for track in tracks:

                    track_id = (
                        track["track_id"]
                    )

                    face = track["face"]

                    x, y, w, h = (
                        face["bbox"]
                    )

                    # ========================================
                    # IDENTITY
                    # ========================================

                    landmark_array = np.array(

                        [
                            face[
                                "landmarks"
                            ][
                                "right_eye"
                            ],

                            face[
                                "landmarks"
                            ][
                                "left_eye"
                            ],

                            face[
                                "landmarks"
                            ][
                                "nose"
                            ],

                            face[
                                "landmarks"
                            ][
                                "right_mouth"
                            ],

                            face[
                                "landmarks"
                            ][
                                "left_mouth"
                            ],
                        ],

                        dtype=np.float32,
                    )

                    aligned_face = (
                        face_align.norm_crop(

                            frame,

                            landmark=(
                                landmark_array
                            ),

                            image_size=112,
                        )
                    )

                    embedding = (

                        recognition_model
                        .get_feat(
                            aligned_face
                        )
                        .flatten()
                    )

                    norm = np.linalg.norm(
                        embedding
                    )

                    if norm > 0:

                        embedding = (
                            embedding / norm
                        )

                    match_result = (
                        find_best_match(

                            embedding,

                            enrolled_embeddings,

                            threshold=(
                                MATCH_THRESHOLD
                            ),
                        )
                    )

                    score = (
                        match_result["score"]
                    )

                    if match_result["matched"]:

                        identity = "RAGAVI"

                    else:

                        identity = "UNKNOWN"

                    # ========================================
                    # HEAD POSE
                    # ========================================

                    if (
                        track_id
                        not in pose_filters
                    ):

                        pose_filters[
                            track_id
                        ] = PoseFilter(

                            window_size=7,

                            alpha=0.15,
                        )

                    raw_pose = (
                        estimate_head_pose(

                            face,

                            frame.shape
                        )
                    )

                    filtered_pose = (
                        pose_filters[
                            track_id
                        ].update(
                            raw_pose
                        )
                    )

                    head_direction = (
                        classify_head_direction(
                            filtered_pose
                        )
                    )

                    yaw = filtered_pose[
                        "yaw"
                    ]

                    pitch = filtered_pose[
                        "pitch"
                    ]

                    roll = filtered_pose[
                        "roll"
                    ]

                    # ========================================
                    # GAZE
                    # ========================================

                    gaze_result = (
                        gaze_estimator.estimate(

                            frame,

                            face
                        )
                    )

                    if gaze_result is None:

                        gaze_direction = (
                            "UNKNOWN"
                        )

                    elif isinstance(
                        gaze_result,
                        dict,
                    ):

                        gaze_direction = (
                            gaze_result.get(
                                "direction",
                                "UNKNOWN",
                            )
                        )

                    else:

                        gaze_direction = str(
                            gaze_result
                        )

                    # ========================================
                    # PROXIMITY
                    # ========================================

                    proximity_result = (
                        proximity_manager.update(

                            track_id,

                            w,
                        )
                    )

                    distance_cm = (
                        proximity_result[
                            "distance_cm"
                        ]
                    )

                    raw_distance_cm = (
                        proximity_result[
                            "raw_distance_cm"
                        ]
                    )

                    proximity = (
                        proximity_result[
                            "proximity"
                        ]
                    )

                    # ========================================
                    # PRIVACY DECISION
                    # ========================================

                    privacy_result = (
                        privacy_engine.decide(

                            identity=identity,

                            distance_cm=(
                                distance_cm
                            ),

                            proximity=(
                                proximity
                            ),

                            head_direction=(
                                head_direction
                            ),

                            gaze_direction=(
                                gaze_direction
                            ),
                        )
                    )

                    decision = (
                        privacy_result[
                            "decision"
                        ]
                    )

                    risk = (
                        privacy_result[
                            "risk"
                        ]
                    )

                    current_decisions.append(

                        {
                            "track_id": track_id,

                            "identity": identity,

                            "score": score,

                            "distance_cm": (
                                distance_cm
                            ),

                            "raw_distance_cm": (
                                raw_distance_cm
                            ),

                            "proximity": (
                                proximity
                            ),

                            "head_direction": (
                                head_direction
                            ),

                            "gaze_direction": (
                                gaze_direction
                            ),

                            "decision": decision,

                            "risk": risk,

                            "bbox": (
                                x,
                                y,
                                w,
                                h,
                            ),

                            "yaw": yaw,

                            "pitch": pitch,

                            "roll": roll,
                        }
                    )

                # ============================================
                # SAVE DECISIONS
                # ============================================

                last_decisions = (
                    current_decisions
                )

            else:

                # No expensive AI processing
                # on this frame.

                faces = []

                tracks = []

            # =================================================
            # DETERMINE MASKING STATE
            # =================================================

            unauthorized_risk = False

            for result in last_decisions:

                if (

                    result["identity"]
                    == "UNKNOWN"

                    and

                    result["decision"]
                    in (
                        "MASK",
                        "BLOCK",
                    )
                ):

                    unauthorized_risk = True

                    break

            # =================================================
            # MASKING STATE CHANGE
            # =================================================

            if (
                unauthorized_risk
                != masking_active
            ):

                masking_active = (
                    unauthorized_risk
                )

                ocr_worker.set_masking(
                    masking_active
                )

                if masking_active:

                    print()
                    print(
                        ">>> UNAUTHORIZED "
                        "VIEWER DETECTED"
                    )

                    print(
                        ">>> SENSITIVE "
                        "MASKING ACTIVE"
                    )

                    for result in (
                        last_decisions
                    ):

                        print(
                            f"    Person "
                            f"{result['track_id']} | "
                            f"{result['identity']} | "
                            f"{result['distance_cm']:.1f} cm | "
                            f"{result['proximity']} | "
                            f"{result['decision']}"
                        )

                    print()

                else:

                    print()
                    print(
                        ">>> PRIVACY ALLOWED"
                    )

                    print(
                        ">>> MASKING DISABLED"
                    )

                    print()

                    # Immediately remove old
                    # overlay.

                    overlay.clear()

                    last_overlay_masks = []

            # =================================================
            # APPLY LATEST OCR MASKS
            # =================================================

            if masking_active:

                masks = (
                    ocr_worker.get_masks()
                )

                # Only update native overlay
                # when OCR result changed.

                if masks != last_overlay_masks:

                    overlay.update_masks(
                        masks
                    )

                    last_overlay_masks = (
                        masks
                    )

                    if masks:

                        detections = (
                            ocr_worker.get_detections()
                        )

                        print(
                            f"[MASKING] "
                            f"{len(masks)} "
                            f"sensitive item(s) masked."
                        )

                        for item in detections:

                            print(
                                f"  "
                                f"{item['type']} "
                                f"("
                                f"{item['confidence']:.0f}%"
                                f")"
                            )

            else:

                if last_overlay_masks:

                    overlay.clear()

                    last_overlay_masks = []

            # =================================================
            # DRAW CAMERA INFORMATION
            # =================================================

            for result in last_decisions:

                x, y, w, h = (
                    result["bbox"]
                )

                identity = (
                    result["identity"]
                )

                score = (
                    result["score"]
                )

                head_direction = (
                    result[
                        "head_direction"
                    ]
                )

                gaze_direction = (
                    result[
                        "gaze_direction"
                    ]
                )

                distance_cm = (
                    result[
                        "distance_cm"
                    ]
                )

                proximity = (
                    result[
                        "proximity"
                    ]
                )

                decision = (
                    result[
                        "decision"
                    ]
                )

                risk = (
                    result["risk"]
                )

                yaw = result["yaw"]

                pitch = result["pitch"]

                # --------------------------------------------
                # FACE BOX
                # --------------------------------------------

                cv2.rectangle(

                    frame,

                    (x, y),

                    (x + w, y + h),

                    (0, 255, 0),

                    2,
                )

                label_x = x

                label_y = max(
                    25,
                    y - 10,
                )

                # --------------------------------------------
                # PERSON
                # --------------------------------------------

                cv2.putText(

                    frame,

                    f"Person "
                    f"{result['track_id']}",

                    (
                        label_x,
                        label_y,
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.65,

                    (0, 255, 255),

                    2,
                )

                # --------------------------------------------
                # IDENTITY
                # --------------------------------------------

                cv2.putText(

                    frame,

                    f"Identity: "
                    f"{identity}",

                    (
                        label_x,
                        y + h + 25,
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.55,

                    (255, 255, 255),

                    2,
                )

                # --------------------------------------------
                # MATCH
                # --------------------------------------------

                cv2.putText(

                    frame,

                    f"Match: "
                    f"{score:.2f}",

                    (
                        label_x,
                        y + h + 47,
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.5,

                    (255, 255, 255),

                    1,
                )

                # --------------------------------------------
                # HEAD
                # --------------------------------------------

                cv2.putText(

                    frame,

                    f"Head: "
                    f"{head_direction}",

                    (
                        label_x,
                        y + h + 69,
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.5,

                    (255, 255, 255),

                    1,
                )

                # --------------------------------------------
                # GAZE
                # --------------------------------------------

                cv2.putText(

                    frame,

                    f"Gaze: "
                    f"{gaze_direction}",

                    (
                        label_x,
                        y + h + 91,
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.5,

                    (255, 255, 255),

                    1,
                )

                # --------------------------------------------
                # DISTANCE
                # --------------------------------------------

                cv2.putText(

                    frame,

                    f"Distance: "
                    f"{distance_cm:.1f} cm",

                    (
                        label_x,
                        y + h + 113,
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.5,

                    (255, 255, 255),

                    1,
                )

                # --------------------------------------------
                # PROXIMITY
                # --------------------------------------------

                cv2.putText(

                    frame,

                    f"Proximity: "
                    f"{proximity}",

                    (
                        label_x,
                        y + h + 135,
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.55,

                    (0, 255, 255),

                    2,
                )

                # --------------------------------------------
                # PRIVACY DECISION
                # --------------------------------------------

                decision_color = (

                    (0, 0, 255)

                    if decision
                    in (
                        "MASK",
                        "BLOCK",
                    )

                    else

                    (0, 255, 0)
                )

                cv2.putText(

                    frame,

                    f"Privacy: "
                    f"{decision}",

                    (
                        label_x,
                        y + h + 157,
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.55,

                    decision_color,

                    2,
                )

                # --------------------------------------------
                # RISK
                # --------------------------------------------

                cv2.putText(

                    frame,

                    f"Risk: "
                    f"{risk:.2f}",

                    (
                        label_x,
                        y + h + 179,
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.5,

                    (255, 255, 255),

                    1,
                )

            # =================================================
            # GLOBAL STATUS
            # =================================================

            if masking_active:

                status = (
                    "UNAUTHORIZED - "
                    "MASKING ACTIVE"
                )

                status_color = (
                    0,
                    0,
                    255,
                )

            else:

                status = (
                    "PRIVACY ALLOWED"
                )

                status_color = (
                    0,
                    255,
                    0,
                )

            cv2.putText(

                frame,

                f"Faces detected: "
                f"{len(last_decisions)}",

                (20, 30),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.65,

                (0, 255, 255),

                2,
            )

            cv2.putText(

                frame,

                f"Active tracks: "
                f"{len(last_decisions)}",

                (20, 58),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.6,

                (0, 255, 255),

                2,
            )

            cv2.putText(

                frame,

                status,

                (20, 90),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.65,

                status_color,

                2,
            )

            # =================================================
            # SHOW CAMERA
            # =================================================

            cv2.imshow(

                "SafeSight - Viewer Detection",

                frame,
            )

            # =================================================
            # QUIT
            # =================================================

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key == ord("q"):

                break

    finally:

        print()
        print(
            "Stopping SafeSight..."
        )

        # ====================================================
        # STOP OCR
        # ====================================================

        try:

            ocr_worker.stop()

        except Exception:
            pass

        # ====================================================
        # CLEAR OVERLAY
        # ====================================================

        try:

            overlay.clear()

        except Exception:
            pass

        # ====================================================
        # CLOSE OVERLAY
        # ====================================================

        try:

            overlay.close()

        except Exception:
            pass

        # ====================================================
        # CAMERA CLEANUP
        # ====================================================

        cap.release()

        cv2.destroyAllWindows()

        # ====================================================
        # GAZE CLEANUP
        # ====================================================

        try:

            gaze_estimator.close()

        except Exception:
            pass

        print(
            "SafeSight Privacy Protection stopped."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()