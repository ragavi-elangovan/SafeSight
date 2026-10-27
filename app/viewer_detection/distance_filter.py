from collections import deque
import numpy as np


class DistanceFilter:
    """
    Temporal smoothing for estimated face distance.

    Uses:
        1. Median filtering to remove sudden spikes.
        2. Exponential moving average for smooth output.
    """

    def __init__(
        self,
        window_size=7,
        alpha=0.20,
    ):
        self.window_size = window_size
        self.alpha = alpha

        self.history = deque(
            maxlen=window_size
        )

        self.smoothed_distance = None

    # ========================================================
    # RESET
    # ========================================================

    def reset(self):
        self.history.clear()
        self.smoothed_distance = None

    # ========================================================
    # UPDATE
    # ========================================================

    def update(self, distance_cm):
        """
        Add a new distance measurement.

        Returns:
            Smoothed distance in cm.
        """

        if distance_cm is None:
            return self.smoothed_distance

        if not np.isfinite(distance_cm):
            return self.smoothed_distance

        if distance_cm <= 0:
            return self.smoothed_distance

        # Add measurement
        self.history.append(
            float(distance_cm)
        )

        # Median filtering
        median_distance = float(
            np.median(
                list(self.history)
            )
        )

        # First measurement
        if self.smoothed_distance is None:

            self.smoothed_distance = (
                median_distance
            )

        else:

            # Exponential moving average
            self.smoothed_distance = (
                self.alpha * median_distance
                + (1.0 - self.alpha)
                * self.smoothed_distance
            )

        return self.smoothed_distance