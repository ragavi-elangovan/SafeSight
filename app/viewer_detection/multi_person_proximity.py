import json

from app.viewer_detection.proximity_estimator import (
    ProximityEstimator,
)

from app.viewer_detection.distance_filter import (
    DistanceFilter,
)


class MultiPersonProximity:
    """
    Maintains independent distance estimation and
    smoothing for every tracked person.

    Calibration is loaded once from a configuration file.
    """

    def __init__(
        self,
        config_path="data/config/proximity_config.json",
    ):

        self.config_path = config_path

        # Load one-time camera calibration
        self.config = self._load_config()

        self.reference_distance_cm = (
            self.config["reference_distance_cm"]
        )

        self.reference_face_width_px = (
            self.config["reference_face_width_px"]
        )

        self.window_size = (
            self.config["distance_filter_window"]
        )

        self.alpha = (
            self.config["distance_filter_alpha"]
        )

        # One estimator per tracked person
        self.estimators = {}

        # One filter per tracked person
        self.filters = {}

    # ========================================================
    # LOAD CONFIGURATION
    # ========================================================

    def _load_config(self):

        try:

            with open(
                self.config_path,
                "r",
                encoding="utf-8",
            ) as file:

                config = json.load(file)

        except FileNotFoundError:

            raise FileNotFoundError(
                f"Proximity configuration not found: "
                f"{self.config_path}\n"
                "Run calibrate_proximity.py first."
            )

        required_keys = [
            "reference_distance_cm",
            "reference_face_width_px",
            "very_close_threshold_cm",
            "close_threshold_cm",
            "medium_threshold_cm",
            "distance_filter_window",
            "distance_filter_alpha",
        ]

        for key in required_keys:

            if key not in config:

                raise ValueError(
                    f"Missing configuration value: {key}"
                )

        return config

    # ========================================================
    # CREATE PERSON STATE
    # ========================================================

    def _create_person(self, track_id):

        self.estimators[track_id] = (
            ProximityEstimator(
                reference_distance_cm=(
                    self.reference_distance_cm
                ),
                reference_face_width_px=(
                    self.reference_face_width_px
                ),
                very_close_threshold_cm=(
                    self.config[
                        "very_close_threshold_cm"
                    ]
                ),
                close_threshold_cm=(
                    self.config[
                        "close_threshold_cm"
                    ]
                ),
                medium_threshold_cm=(
                    self.config[
                        "medium_threshold_cm"
                    ]
                ),
            )
        )

        self.filters[track_id] = (
            DistanceFilter(
                window_size=self.window_size,
                alpha=self.alpha,
            )
        )

    # ========================================================
    # UPDATE PERSON
    # ========================================================

    def update(
        self,
        track_id,
        face_width_px,
    ):

        if track_id not in self.estimators:

            self._create_person(
                track_id
            )

        estimator = self.estimators[
            track_id
        ]

        distance_filter = self.filters[
            track_id
        ]

        # ----------------------------------------------------
        # RAW DISTANCE
        # ----------------------------------------------------

        raw_distance = (
            estimator.estimate_distance(
                face_width_px
            )
        )

        # ----------------------------------------------------
        # SMOOTH DISTANCE
        # ----------------------------------------------------

        stable_distance = (
            distance_filter.update(
                raw_distance
            )
        )

        # ----------------------------------------------------
        # PROXIMITY
        # ----------------------------------------------------

        proximity = (
            estimator.classify(
                stable_distance
            )
        )

        return {
            "track_id": track_id,
            "face_width_px": float(
                face_width_px
            ),
            "raw_distance_cm": raw_distance,
            "distance_cm": stable_distance,
            "proximity": proximity,
        }

    # ========================================================
    # REMOVE PERSON
    # ========================================================

    def remove(self, track_id):

        self.estimators.pop(
            track_id,
            None,
        )

        self.filters.pop(
            track_id,
            None,
        )

    # ========================================================
    # CLEANUP
    # ========================================================

    def cleanup(self, active_track_ids):

        active_track_ids = set(
            active_track_ids
        )

        existing_ids = set(
            self.estimators.keys()
        )

        inactive_ids = (
            existing_ids - active_track_ids
        )

        for track_id in inactive_ids:

            self.remove(
                track_id
            )

    # ========================================================
    # ACTIVE IDS
    # ========================================================

    def active_ids(self):

        return sorted(
            self.estimators.keys()
        )