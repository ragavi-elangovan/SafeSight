import math

from app.viewer_detection.head_pose import (
    estimate_head_pose,
    PoseFilter,
)


class PoseTrack:
    """
    Stores the state of one tracked face.
    """

    def __init__(
        self,
        track_id,
        face_info,
        frame_shape,
        window_size=7,
        alpha=0.15,
    ):
        self.track_id = track_id

        self.last_bbox = face_info["bbox"]

        self.missing_frames = 0

        self.pose_filter = PoseFilter(
            window_size=window_size,
            alpha=alpha,
        )

        raw_pose = estimate_head_pose(
            face_info,
            frame_shape,
        )

        self.pose = self.pose_filter.update(
            raw_pose
        )

    def update(
        self,
        face_info,
        frame_shape,
    ):
        """
        Update this track with the latest face detection.
        """

        self.last_bbox = face_info["bbox"]

        self.missing_frames = 0

        raw_pose = estimate_head_pose(
            face_info,
            frame_shape,
        )

        self.pose = self.pose_filter.update(
            raw_pose
        )

        return self.pose

    def mark_missing(self):
        """
        Mark this track as temporarily missing.
        """

        self.missing_frames += 1


class MultiPersonPoseTracker:
    """
    Lightweight multi-person face tracker.

    Each currently tracked face receives a small ID:

        Person 1
        Person 2
        Person 3

    IDs are reused when old tracks disappear.

    Tracking is based on:
        1. Bounding-box center distance
        2. Bounding-box IoU
    """

    def __init__(
        self,
        max_distance=250.0,
        max_missing_frames=8,
        window_size=7,
        alpha=0.15,
    ):
        self.max_distance = max_distance

        self.max_missing_frames = (
            max_missing_frames
        )

        self.window_size = window_size

        self.alpha = alpha

        # Active tracks
        self.tracks = {}

    # =====================================================
    # BOUNDING BOX CENTER
    # =====================================================

    @staticmethod
    def bbox_center(bbox):
        """
        Return center of bounding box.
        """

        x, y, w, h = bbox

        return (
            x + w / 2.0,
            y + h / 2.0,
        )

    # =====================================================
    # CENTER DISTANCE
    # =====================================================

    @staticmethod
    def center_distance(
        center_a,
        center_b,
    ):
        """
        Euclidean distance between two points.
        """

        dx = (
            center_a[0]
            - center_b[0]
        )

        dy = (
            center_a[1]
            - center_b[1]
        )

        return math.sqrt(
            dx * dx + dy * dy
        )

    # =====================================================
    # IOU
    # =====================================================

    @staticmethod
    def calculate_iou(
        bbox_a,
        bbox_b,
    ):
        """
        Calculate Intersection over Union
        between two bounding boxes.
        """

        ax, ay, aw, ah = bbox_a

        bx, by, bw, bh = bbox_b

        ax2 = ax + aw
        ay2 = ay + ah

        bx2 = bx + bw
        by2 = by + bh

        intersection_x1 = max(
            ax,
            bx,
        )

        intersection_y1 = max(
            ay,
            by,
        )

        intersection_x2 = min(
            ax2,
            bx2,
        )

        intersection_y2 = min(
            ay2,
            by2,
        )

        intersection_width = max(
            0,
            intersection_x2
            - intersection_x1,
        )

        intersection_height = max(
            0,
            intersection_y2
            - intersection_y1,
        )

        intersection_area = (
            intersection_width
            * intersection_height
        )

        area_a = aw * ah
        area_b = bw * bh

        union_area = (
            area_a
            + area_b
            - intersection_area
        )

        if union_area <= 0:
            return 0.0

        return (
            intersection_area
            / union_area
        )

    # =====================================================
    # GET LOWEST AVAILABLE ID
    # =====================================================

    def _get_available_id(self):
        """
        Return the smallest available Person ID.

        Example:

            Existing: 1, 2
            New ID:   3

            Existing: 1, 3
            New ID:   2
        """

        track_id = 1

        while track_id in self.tracks:
            track_id += 1

        return track_id

    # =====================================================
    # CREATE TRACK
    # =====================================================

    def _create_track(
        self,
        face_info,
        frame_shape,
    ):
        """
        Create a new pose track.
        """

        track_id = (
            self._get_available_id()
        )

        track = PoseTrack(
            track_id=track_id,
            face_info=face_info,
            frame_shape=frame_shape,
            window_size=self.window_size,
            alpha=self.alpha,
        )

        self.tracks[
            track_id
        ] = track

        return track

    # =====================================================
    # UPDATE TRACKER
    # =====================================================

    def update(
        self,
        faces,
        frame_shape,
    ):
        """
        Update all currently detected faces.

        Returns:

        [
            {
                "track_id": 1,
                "face": face_info,
                "pose": pose
            },
            ...
        ]
        """

        # -------------------------------------------------
        # No faces detected
        # -------------------------------------------------

        if len(faces) == 0:

            for track in (
                self.tracks.values()
            ):
                track.mark_missing()

            self._remove_old_tracks()

            return []

        # -------------------------------------------------
        # No existing tracks
        # -------------------------------------------------

        if len(self.tracks) == 0:

            results = []

            for face_info in faces:

                track = (
                    self._create_track(
                        face_info,
                        frame_shape,
                    )
                )

                results.append(
                    {
                        "track_id": (
                            track.track_id
                        ),
                        "face": face_info,
                        "pose": track.pose,
                    }
                )

            return results

        # -------------------------------------------------
        # Existing tracks
        # -------------------------------------------------

        track_items = list(
            self.tracks.items()
        )

        candidates = []

        # -------------------------------------------------
        # Compare every detection with every track
        # -------------------------------------------------

        for (
            face_index,
            face_info,
        ) in enumerate(faces):

            detection_bbox = (
                face_info["bbox"]
            )

            detection_center = (
                self.bbox_center(
                    detection_bbox
                )
            )

            for (
                track_id,
                track,
            ) in track_items:

                track_center = (
                    self.bbox_center(
                        track.last_bbox
                    )
                )

                distance = (
                    self.center_distance(
                        detection_center,
                        track_center,
                    )
                )

                iou = (
                    self.calculate_iou(
                        detection_bbox,
                        track.last_bbox,
                    )
                )

                # -------------------------------------------------
                # Accept possible match if:
                #
                # center is close
                #
                # OR
                #
                # bounding boxes overlap
                # -------------------------------------------------

                if (
                    distance
                    <= self.max_distance
                    or iou >= 0.15
                ):

                    # Lower score = better match.
                    #
                    # Give overlapping boxes a
                    # small advantage.
                    score = distance

                    if iou > 0:

                        score -= (
                            iou * 150.0
                        )

                    candidates.append(
                        (
                            score,
                            distance,
                            iou,
                            face_index,
                            track_id,
                        )
                    )

        # -------------------------------------------------
        # Best matches first
        # -------------------------------------------------

        candidates.sort(
            key=lambda item: item[0]
        )

        matched_faces = set()

        matched_tracks = set()

        assignments = []

        # -------------------------------------------------
        # Match detections to tracks
        # -------------------------------------------------

        for (
            score,
            distance,
            iou,
            face_index,
            track_id,
        ) in candidates:

            if (
                face_index
                in matched_faces
            ):
                continue

            if (
                track_id
                in matched_tracks
            ):
                continue

            matched_faces.add(
                face_index
            )

            matched_tracks.add(
                track_id
            )

            assignments.append(
                (
                    face_index,
                    track_id,
                )
            )

        # -------------------------------------------------
        # Mark unmatched tracks
        # -------------------------------------------------

        for (
            track_id,
            track,
        ) in track_items:

            if (
                track_id
                not in matched_tracks
            ):

                track.mark_missing()

        # -------------------------------------------------
        # Update matched tracks
        # -------------------------------------------------

        results = []

        for (
            face_index,
            track_id,
        ) in assignments:

            face_info = faces[
                face_index
            ]

            track = self.tracks[
                track_id
            ]

            pose = track.update(
                face_info,
                frame_shape,
            )

            results.append(
                {
                    "track_id": track_id,
                    "face": face_info,
                    "pose": pose,
                }
            )

        # -------------------------------------------------
        # Create tracks for new faces
        # -------------------------------------------------

        for (
            face_index,
            face_info,
        ) in enumerate(faces):

            if (
                face_index
                in matched_faces
            ):
                continue

            track = (
                self._create_track(
                    face_info,
                    frame_shape,
                )
            )

            results.append(
                {
                    "track_id": (
                        track.track_id
                    ),
                    "face": face_info,
                    "pose": track.pose,
                }
            )

        # -------------------------------------------------
        # Remove expired tracks
        # -------------------------------------------------

        self._remove_old_tracks()

        # -------------------------------------------------
        # Sort by Person ID
        # -------------------------------------------------

        results.sort(
            key=lambda item:
                item["track_id"]
        )

        return results

    # =====================================================
    # REMOVE OLD TRACKS
    # =====================================================

    def _remove_old_tracks(self):
        """
        Delete tracks that have been missing
        for too many frames.
        """

        expired_ids = []

        for (
            track_id,
            track,
        ) in self.tracks.items():

            if (
                track.missing_frames
                > self.max_missing_frames
            ):

                expired_ids.append(
                    track_id
                )

        for track_id in expired_ids:

            del self.tracks[
                track_id
            ]

    # =====================================================
    # RESET
    # =====================================================

    def reset(self):
        """
        Clear all active tracks.
        """

        self.tracks.clear()