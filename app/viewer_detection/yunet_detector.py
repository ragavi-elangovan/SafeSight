import cv2
import numpy as np
from pathlib import Path


class YuNetDetector:
    def __init__(
        self,
        model_path: str,
        input_size=(320, 320),
        score_threshold=0.6,
        nms_threshold=0.3,
        top_k=5000,
    ):
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"YuNet model not found: {model_path}"
            )

        self.detector = cv2.FaceDetectorYN.create(
            str(model_path),
            "",
            input_size,
            score_threshold,
            nms_threshold,
            top_k,
        )

    def detect(self, frame: np.ndarray):
        height, width = frame.shape[:2]

        self.detector.setInputSize((width, height))

        _, faces = self.detector.detect(frame)

        if faces is None:
            return []

        results = []

        for face in faces:
            x, y, w, h = face[:4]

            landmarks = {
                "right_eye": (float(face[4]), float(face[5])),
                "left_eye": (float(face[6]), float(face[7])),
                "nose": (float(face[8]), float(face[9])),
                "right_mouth": (float(face[10]), float(face[11])),
                "left_mouth": (float(face[12]), float(face[13])),
            }

            results.append(
                {
                    "bbox": (int(x), int(y), int(w), int(h)),
                    "landmarks": landmarks,
                    "confidence": float(face[14]),
                }
            )

        return results