import cv2

from app.viewer_detection.sensitive_info_detector import (
    SensitiveInfoDetector,
    draw_sensitive_detections
)


# ============================================================
# IMAGE PATH
# ============================================================

IMAGE_PATH = "test_sensitive_screen.png"


# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(IMAGE_PATH)

if image is None:

    raise FileNotFoundError(
        f"Could not load image: {IMAGE_PATH}"
    )


# ============================================================
# CREATE DETECTOR
# ============================================================

detector = SensitiveInfoDetector()


# ============================================================
# DETECT
# ============================================================

detections = detector.detect(image)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 70)
print("SafeSight - SENSITIVE INFORMATION DETECTOR")
print("=" * 70)

print(
    f"\nSensitive items detected: {len(detections)}"
)


for detection in detections:

    print()
    print(
        f"Type       : {detection['type']}"
    )

    print(
        f"Text       : {detection['text']}"
    )

    print(
        f"Confidence : "
        f"{detection['confidence']:.2f}%"
    )

    print(
        f"BoundingBox: "
        f"{detection['bbox']}"
    )


# ============================================================
# DRAW RESULTS
# ============================================================

output = draw_sensitive_detections(
    image,
    detections
)


# ============================================================
# SAVE RESULT
# ============================================================

OUTPUT_PATH = "test_sensitive_screen_detected.png"

cv2.imwrite(
    OUTPUT_PATH,
    output
)


print()
print(
    f"Result saved to: {OUTPUT_PATH}"
)

print()
print("=" * 70)
print("Sensitive information detection completed.")
print("=" * 70)