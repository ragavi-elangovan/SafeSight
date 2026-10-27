import cv2
import numpy as np


# =========================================================
# 3D FACE MODEL
# =========================================================
#
# Order must match YuNet landmarks:
#
# 0 -> right eye
# 1 -> left eye
# 2 -> nose
# 3 -> right mouth
# 4 -> left mouth
#
# These are approximate 3D coordinates used for the
# initial head-pose prototype.
# =========================================================

FACE_3D_MODEL = np.array(
    [
        [-30.0, -30.0, 0.0],
        [30.0, -30.0, 0.0],
        [0.0, 0.0, -5.0],
        [-25.0, 30.0, 0.0],
        [25.0, 30.0, 0.0],
    ],
    dtype=np.float64,
)


# =========================================================
# LANDMARK EXTRACTION
# =========================================================

def get_landmark_array(face_info):
    """
    Convert YuNet landmarks into a 5x2 NumPy array.
    """

    landmarks = face_info["landmarks"]

    points = np.array(
        [
            landmarks["right_eye"],
            landmarks["left_eye"],
            landmarks["nose"],
            landmarks["right_mouth"],
            landmarks["left_mouth"],
        ],
        dtype=np.float64,
    )

    return points


# =========================================================
# HEAD POSE ESTIMATION
# =========================================================

def estimate_head_pose(face_info, frame_shape):
    """
    Estimate head pose using YuNet's five landmarks.

    Returns:
        {
            "yaw": float,
            "pitch": float,
            "roll": float
        }

    Values are in degrees.
    """

    image_points = get_landmark_array(face_info)

    if image_points.shape != (5, 2):
        return None

    if not np.isfinite(image_points).all():
        return None

    height, width = frame_shape[:2]

    # -----------------------------------------------------
    # Approximate camera matrix
    # -----------------------------------------------------

    focal_length = float(max(width, height))

    center = (
        width / 2.0,
        height / 2.0,
    )

    camera_matrix = np.array(
        [
            [focal_length, 0.0, center[0]],
            [0.0, focal_length, center[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    distortion_coefficients = np.zeros(
        (4, 1),
        dtype=np.float64,
    )

    # -----------------------------------------------------
    # Solve PnP
    # -----------------------------------------------------

    try:

        success, rotation_vector, translation_vector = cv2.solvePnP(
            FACE_3D_MODEL,
            image_points,
            camera_matrix,
            distortion_coefficients,
            flags=cv2.SOLVEPNP_SQPNP,
        )

    except cv2.error:
        return None

    if not success:
        return None

    # -----------------------------------------------------
    # Rotation vector -> rotation matrix
    # -----------------------------------------------------

    rotation_matrix, _ = cv2.Rodrigues(
        rotation_vector
    )

    # -----------------------------------------------------
    # Rotation matrix -> Euler angles
    # -----------------------------------------------------

    angles, _, _, _, _, _ = cv2.RQDecomp3x3(
        rotation_matrix
    )

    pitch = float(angles[0])
    yaw = float(angles[1])
    roll = float(angles[2])

    return {
        "yaw": yaw,
        "pitch": pitch,
        "roll": roll,
    }


# =========================================================
# MEDIAN POSE FILTER
# =========================================================

class PoseFilter:
    """
    Two-stage temporal filter.

    Stage 1:
        Median filter removes sudden landmark/PnP spikes.

    Stage 2:
        Exponential moving average provides smooth movement.
    """

    def __init__(
        self,
        window_size=7,
        alpha=0.15,
    ):
        self.window_size = window_size
        self.alpha = alpha

        self.yaw_history = []
        self.pitch_history = []
        self.roll_history = []

        self.smoothed_pose = None

    def reset(self):
        """
        Reset filter state.
        """

        self.yaw_history.clear()
        self.pitch_history.clear()
        self.roll_history.clear()

        self.smoothed_pose = None

    def update(self, pose):
        """
        Add a new pose and return the filtered pose.
        """

        if pose is None:
            return self.smoothed_pose

        # -------------------------------------------------
        # Add raw values
        # -------------------------------------------------

        self.yaw_history.append(pose["yaw"])
        self.pitch_history.append(pose["pitch"])
        self.roll_history.append(pose["roll"])

        # Keep only the latest N values
        if len(self.yaw_history) > self.window_size:
            self.yaw_history.pop(0)

        if len(self.pitch_history) > self.window_size:
            self.pitch_history.pop(0)

        if len(self.roll_history) > self.window_size:
            self.roll_history.pop(0)

        # -------------------------------------------------
        # Median filter
        # -------------------------------------------------

        median_yaw = float(
            np.median(self.yaw_history)
        )

        median_pitch = float(
            np.median(self.pitch_history)
        )

        median_roll = float(
            np.median(self.roll_history)
        )

        median_pose = {
            "yaw": median_yaw,
            "pitch": median_pitch,
            "roll": median_roll,
        }

        # -------------------------------------------------
        # First value
        # -------------------------------------------------

        if self.smoothed_pose is None:

            self.smoothed_pose = median_pose.copy()

            return self.smoothed_pose

        # -------------------------------------------------
        # Exponential smoothing
        # -------------------------------------------------

        self.smoothed_pose = {
            "yaw": (
                self.alpha * median_pose["yaw"]
                + (1.0 - self.alpha)
                * self.smoothed_pose["yaw"]
            ),

            "pitch": (
                self.alpha * median_pose["pitch"]
                + (1.0 - self.alpha)
                * self.smoothed_pose["pitch"]
            ),

            "roll": (
                self.alpha * median_pose["roll"]
                + (1.0 - self.alpha)
                * self.smoothed_pose["roll"]
            ),
        }

        return self.smoothed_pose


# =========================================================
# SIMPLE DIRECTION CLASSIFICATION
# =========================================================

def classify_head_direction(
    pose,
    yaw_threshold=18.0,
    pitch_threshold=18.0,
):
    """
    Classify head orientation.

    Returns:

        FORWARD
        LEFT
        RIGHT
        UP
        DOWN
        UP-LEFT
        UP-RIGHT
        DOWN-LEFT
        DOWN-RIGHT
    """

    if pose is None:
        return "UNKNOWN"

    yaw = pose["yaw"]
    pitch = pose["pitch"]

    # -----------------------------------------------------
    # Horizontal
    # -----------------------------------------------------

    if yaw > yaw_threshold:
        horizontal = "LEFT"

    elif yaw < -yaw_threshold:
        horizontal = "RIGHT"

    else:
        horizontal = "CENTER"

    # -----------------------------------------------------
    # Vertical
    # -----------------------------------------------------

    if pitch > pitch_threshold:
        vertical = "DOWN"

    elif pitch < -pitch_threshold:
        vertical = "UP"

    else:
        vertical = "CENTER"

    # -----------------------------------------------------
    # Final direction
    # -----------------------------------------------------

    if (
        horizontal == "CENTER"
        and vertical == "CENTER"
    ):
        return "FORWARD"

    if horizontal == "CENTER":
        return vertical

    if vertical == "CENTER":
        return horizontal

    return f"{vertical}-{horizontal}"


# =========================================================
# DRAW HEAD POSE
# =========================================================

def draw_head_pose(
    frame,
    face_info,
    pose,
):
    """
    Draw filtered head-pose values.
    """

    if pose is None:
        return

    x, y, w, h = face_info["bbox"]

    yaw = pose["yaw"]
    pitch = pose["pitch"]
    roll = pose["roll"]

    direction = classify_head_direction(pose)

    text_x = x
    text_y = y + h + 45

    # -----------------------------------------------------
    # Yaw
    # -----------------------------------------------------

    cv2.putText(
        frame,
        f"Yaw: {yaw:.1f}",
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 0),
        1,
    )

    # -----------------------------------------------------
    # Pitch
    # -----------------------------------------------------

    cv2.putText(
        frame,
        f"Pitch: {pitch:.1f}",
        (text_x, text_y + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 0),
        1,
    )

    # -----------------------------------------------------
    # Roll
    # -----------------------------------------------------

    cv2.putText(
        frame,
        f"Roll: {roll:.1f}",
        (text_x, text_y + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 0),
        1,
    )

    # -----------------------------------------------------
    # Direction
    # -----------------------------------------------------

    cv2.putText(
        frame,
        direction,
        (text_x, text_y + 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
    )