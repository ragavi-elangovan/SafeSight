import cv2
import numpy as np
import mediapipe as mp


class GazeEstimator:
    """
    Estimate approximate eye gaze direction using
    MediaPipe Face Mesh iris landmarks.

    Output:
        CENTER
        LEFT
        RIGHT
        UP
        DOWN
    """

    def __init__(
        self,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):

        self.mp_face_mesh = (
            mp.solutions.face_mesh
        )

        self.face_mesh = (
            self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=(
                    min_detection_confidence
                ),
                min_tracking_confidence=(
                    min_tracking_confidence
                ),
            )
        )

    # =====================================================
    # LANDMARK HELPERS
    # =====================================================

    @staticmethod
    def _point(
        landmarks,
        index,
        width,
        height,
    ):
        """
        Convert normalized MediaPipe landmark
        into pixel coordinates.
        """

        landmark = landmarks[index]

        return np.array(
            [
                landmark.x * width,
                landmark.y * height,
            ],
            dtype=np.float32,
        )

    # =====================================================
    # EYE RATIO
    # =====================================================

    @staticmethod
    def _eye_ratio(
        iris,
        corner_a,
        corner_b,
    ):
        """
        Calculate horizontal iris position
        between two eye corners.

        0.0 -> one side
        0.5 -> center
        1.0 -> other side
        """

        eye_width = np.linalg.norm(
            corner_b - corner_a
        )

        if eye_width < 1.0:
            return 0.5

        projection = np.dot(
            iris - corner_a,
            corner_b - corner_a,
        )

        denominator = np.dot(
            corner_b - corner_a,
            corner_b - corner_a,
        )

        if denominator <= 0:
            return 0.5

        ratio = (
            projection
            / denominator
        )

        return float(
            np.clip(
                ratio,
                0.0,
                1.0,
            )
        )

    # =====================================================
    # VERTICAL EYE RATIO
    # =====================================================

    @staticmethod
    def _vertical_ratio(
        iris,
        upper,
        lower,
    ):
        """
        Calculate vertical iris position
        between upper and lower eyelids.
        """

        eye_height = np.linalg.norm(
            lower - upper
        )

        if eye_height < 1.0:
            return 0.5

        projection = np.dot(
            iris - upper,
            lower - upper,
        )

        denominator = np.dot(
            lower - upper,
            lower - upper,
        )

        if denominator <= 0:
            return 0.5

        ratio = (
            projection
            / denominator
        )

        return float(
            np.clip(
                ratio,
                0.0,
                1.0,
            )
        )

    # =====================================================
    # PROCESS ONE FACE
    # =====================================================

    def estimate(
        self,
        frame,
        face_info,
    ):
        """
        Estimate gaze for one YuNet face.

        Returns:

        {
            "horizontal": float,
            "vertical": float,
            "direction": str
        }

        or None if eye landmarks aren't available.
        """

        x, y, w, h = face_info["bbox"]

        frame_height, frame_width = (
            frame.shape[:2]
        )

        # -------------------------------------------------
        # Add padding around face
        # -------------------------------------------------

        padding_x = int(
            w * 0.20
        )

        padding_y = int(
            h * 0.20
        )

        x1 = max(
            0,
            x - padding_x,
        )

        y1 = max(
            0,
            y - padding_y,
        )

        x2 = min(
            frame_width,
            x + w + padding_x,
        )

        y2 = min(
            frame_height,
            y + h + padding_y,
        )

        face_crop = frame[
            y1:y2,
            x1:x2,
        ]

        if face_crop.size == 0:
            return None

        crop_height, crop_width = (
            face_crop.shape[:2]
        )

        # -------------------------------------------------
        # BGR -> RGB
        # -------------------------------------------------

        rgb = cv2.cvtColor(
            face_crop,
            cv2.COLOR_BGR2RGB,
        )

        rgb.flags.writeable = False

        results = self.face_mesh.process(
            rgb
        )

        if not results.multi_face_landmarks:
            return None

        mesh = (
            results.multi_face_landmarks[0]
        )

        landmarks = mesh.landmark

        # -------------------------------------------------
        # MediaPipe iris landmarks
        #
        # Right iris:
        #   469,470,471,472
        #
        # Left iris:
        #   474,475,476,477
        #
        # Center:
        #   468 = right iris center
        #   473 = left iris center
        # -------------------------------------------------

        right_iris = self._point(
            landmarks,
            468,
            crop_width,
            crop_height,
        )

        left_iris = self._point(
            landmarks,
            473,
            crop_width,
            crop_height,
        )

        # -------------------------------------------------
        # Right eye corners
        # -------------------------------------------------

        right_corner_a = self._point(
            landmarks,
            33,
            crop_width,
            crop_height,
        )

        right_corner_b = self._point(
            landmarks,
            133,
            crop_width,
            crop_height,
        )

        # -------------------------------------------------
        # Left eye corners
        # -------------------------------------------------

        left_corner_a = self._point(
            landmarks,
            362,
            crop_width,
            crop_height,
        )

        left_corner_b = self._point(
            landmarks,
            263,
            crop_width,
            crop_height,
        )

        # -------------------------------------------------
        # Right eye vertical
        # -------------------------------------------------

        right_upper = self._point(
            landmarks,
            159,
            crop_width,
            crop_height,
        )

        right_lower = self._point(
            landmarks,
            145,
            crop_width,
            crop_height,
        )

        # -------------------------------------------------
        # Left eye vertical
        # -------------------------------------------------

        left_upper = self._point(
            landmarks,
            386,
            crop_width,
            crop_height,
        )

        left_lower = self._point(
            landmarks,
            374,
            crop_width,
            crop_height,
        )

        # -------------------------------------------------
        # Calculate horizontal ratios
        # -------------------------------------------------

        right_horizontal = (
            self._eye_ratio(
                right_iris,
                right_corner_a,
                right_corner_b,
            )
        )

        left_horizontal = (
            self._eye_ratio(
                left_iris,
                left_corner_a,
                left_corner_b,
            )
        )

        # -------------------------------------------------
        # Calculate vertical ratios
        # -------------------------------------------------

        right_vertical = (
            self._vertical_ratio(
                right_iris,
                right_upper,
                right_lower,
            )
        )

        left_vertical = (
            self._vertical_ratio(
                left_iris,
                left_upper,
                left_lower,
            )
        )

        # -------------------------------------------------
        # Average both eyes
        # -------------------------------------------------

        horizontal = (
            right_horizontal
            + left_horizontal
        ) / 2.0

        vertical = (
            right_vertical
            + left_vertical
        ) / 2.0

        # -------------------------------------------------
        # Classify gaze
        # -------------------------------------------------

        direction = self.classify_gaze(
            horizontal,
            vertical,
        )

        return {
            "horizontal": horizontal,
            "vertical": vertical,
            "direction": direction,
        }

    # =====================================================
    # CLASSIFY GAZE
    # =====================================================

    @staticmethod
    def classify_gaze(
        horizontal,
        vertical,
        horizontal_threshold=0.12,
        vertical_threshold=0.12,
    ):
        """
        Classify normalized gaze position.

        Center is approximately 0.5.
        """

        horizontal_offset = (
            horizontal - 0.5
        )

        vertical_offset = (
            vertical - 0.5
        )

        # -------------------------------------------------
        # Horizontal
        # -------------------------------------------------

        if (
            horizontal_offset
            > horizontal_threshold
        ):
            horizontal_direction = "RIGHT"

        elif (
            horizontal_offset
            < -horizontal_threshold
        ):
            horizontal_direction = "LEFT"

        else:
            horizontal_direction = "CENTER"

        # -------------------------------------------------
        # Vertical
        # -------------------------------------------------

        if (
            vertical_offset
            > vertical_threshold
        ):
            vertical_direction = "DOWN"

        elif (
            vertical_offset
            < -vertical_threshold
        ):
            vertical_direction = "UP"

        else:
            vertical_direction = "CENTER"

        # -------------------------------------------------
        # Combine
        # -------------------------------------------------

        if (
            horizontal_direction
            == "CENTER"
            and vertical_direction
            == "CENTER"
        ):
            return "CENTER"

        if horizontal_direction == "CENTER":
            return vertical_direction

        if vertical_direction == "CENTER":
            return horizontal_direction

        return (
            f"{vertical_direction}-"
            f"{horizontal_direction}"
        )

    # =====================================================
    # RELEASE
    # =====================================================

    def close(self):
        """
        Release MediaPipe resources.
        """

        self.face_mesh.close()