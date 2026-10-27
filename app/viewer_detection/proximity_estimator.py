import json


class ProximityEstimator:

    def __init__(
        self,
        reference_distance_cm=60.0,
        reference_face_width_px=None,
        very_close_threshold_cm=40.0,
        close_threshold_cm=70.0,
        medium_threshold_cm=120.0,
    ):

        self.reference_distance_cm = (
            float(reference_distance_cm)
        )

        self.reference_face_width_px = (
            reference_face_width_px
        )

        self.very_close_threshold_cm = (
            float(very_close_threshold_cm)
        )

        self.close_threshold_cm = (
            float(close_threshold_cm)
        )

        self.medium_threshold_cm = (
            float(medium_threshold_cm)
        )

    # ========================================================
    # LOAD CONFIGURATION
    # ========================================================

    @classmethod
    def from_config(cls, config_path):

        with open(
            config_path,
            "r",
            encoding="utf-8",
        ) as file:

            config = json.load(file)

        return cls(
            reference_distance_cm=config[
                "reference_distance_cm"
            ],

            reference_face_width_px=config[
                "reference_face_width_px"
            ],

            very_close_threshold_cm=config[
                "very_close_threshold_cm"
            ],

            close_threshold_cm=config[
                "close_threshold_cm"
            ],

            medium_threshold_cm=config[
                "medium_threshold_cm"
            ],
        )

    # ========================================================
    # CALIBRATE
    # ========================================================

    def calibrate(
        self,
        face_width_px,
        distance_cm=None,
    ):

        if distance_cm is not None:

            self.reference_distance_cm = (
                float(distance_cm)
            )

        if face_width_px <= 0:

            raise ValueError(
                "Face width must be greater than zero."
            )

        self.reference_face_width_px = (
            float(face_width_px)
        )

    # ========================================================
    # DISTANCE
    # ========================================================

    def estimate_distance(
        self,
        face_width_px,
    ):

        if (
            self.reference_face_width_px
            is None
        ):
            return None

        if face_width_px <= 0:
            return None

        distance = (
            self.reference_distance_cm
            * self.reference_face_width_px
            / face_width_px
        )

        return float(distance)

    # ========================================================
    # CLASSIFICATION
    # ========================================================

    def classify(
        self,
        distance_cm,
    ):

        if distance_cm is None:

            return "UNKNOWN"

        if distance_cm < (
            self.very_close_threshold_cm
        ):

            return "VERY CLOSE"

        elif distance_cm < (
            self.close_threshold_cm
        ):

            return "CLOSE"

        elif distance_cm < (
            self.medium_threshold_cm
        ):

            return "MEDIUM"

        else:

            return "FAR"

    # ========================================================
    # COMPLETE ESTIMATION
    # ========================================================

    def estimate(
        self,
        face_width_px,
    ):

        distance = self.estimate_distance(
            face_width_px
        )

        proximity = self.classify(
            distance
        )

        return {
            "distance_cm": distance,
            "proximity": proximity,
        }