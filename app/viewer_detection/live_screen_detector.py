import time
import mss
import numpy as np

from app.viewer_detection.sensitive_info_detector import SensitiveInfoDetector
from app.viewer_detection.screen_mask_overlay import ScreenMaskOverlay


def main():
    print("=" * 60)
    print("SafeSight - LIVE SENSITIVE INFORMATION MASKING")
    print("=" * 60)

    # =========================================================
    # 1. GET SCREEN INFORMATION
    # =========================================================
    with mss.MSS() as sct:
        monitor = sct.monitors[1]

        screen_x = monitor["left"]
        screen_y = monitor["top"]
        screen_width = monitor["width"]
        screen_height = monitor["height"]

    print(f"Screen: {screen_width}x{screen_height}")
    print(f"Position: ({screen_x}, {screen_y})")

    # =========================================================
    # 2. CREATE SENSITIVE INFORMATION DETECTOR
    # =========================================================
    detector = SensitiveInfoDetector()

    # =========================================================
    # 3. CREATE NATIVE WINDOWS OVERLAY
    # =========================================================
    overlay = ScreenMaskOverlay(
        screen_width=screen_width,
        screen_height=screen_height,
        screen_x=screen_x,
        screen_y=screen_y
    )

    overlay.show()

    print()
    print("Live masking started.")
    print("Sensitive information will be masked directly on your screen.")
    print("Press CTRL+C to stop.")
    print()

    # =========================================================
    # 4. OCR INTERVAL
    # =========================================================
    last_detection_time = 0
    detection_interval = 0.5

    try:

        with mss.MSS() as sct:

            while True:

                current_time = time.time()

                # -------------------------------------------------
                # Run OCR only every 0.5 seconds
                # -------------------------------------------------
                if current_time - last_detection_time < detection_interval:
                    time.sleep(0.05)
                    continue

                last_detection_time = current_time

                # -------------------------------------------------
                # Capture the real desktop
                # -------------------------------------------------
                screenshot = np.array(
                    sct.grab(monitor)
                )

                # MSS returns BGRA
                # Remove alpha channel
                frame = screenshot[:, :, :3]

                # -------------------------------------------------
                # Detect sensitive information
                # -------------------------------------------------
                detections = detector.detect(frame)

                # -------------------------------------------------
                # Prepare overlay masks
                # -------------------------------------------------
                masks = []

                for detection in detections:

                    bbox = detection.get("bbox")

                    if bbox is None:
                        continue

                    x, y, w, h = bbox

                    # -------------------------------------------------
                    # Add small padding around detected text
                    # -------------------------------------------------
                    padding = 5

                    x1 = max(
                        0,
                        int(x) - padding
                    )

                    y1 = max(
                        0,
                        int(y) - padding
                    )

                    x2 = min(
                        screen_width,
                        int(x + w) + padding
                    )

                    y2 = min(
                        screen_height,
                        int(y + h) + padding
                    )

                    # -------------------------------------------------
                    # IMPORTANT:
                    # ScreenMaskOverlay.update_masks()
                    # expects a "bbox" key.
                    # -------------------------------------------------
                    masks.append(
                        {
                            "bbox": (
                                x1,
                                y1,
                                x2 - x1,
                                y2 - y1
                            )
                        }
                    )

                # -------------------------------------------------
                # Update the native Windows overlay
                # -------------------------------------------------
                overlay.update_masks(masks)

                # -------------------------------------------------
                # Print detection information
                # -------------------------------------------------
                if detections:

                    print(
                        f"Detected {len(detections)} "
                        f"sensitive item(s):"
                    )

                    for detection in detections:

                        info_type = detection.get(
                            "type",
                            "UNKNOWN"
                        )

                        confidence = detection.get(
                            "confidence",
                            0
                        )

                        print(
                            f"  {info_type}: "
                            f"{confidence:.2f}"
                        )

                else:

                    print(
                        "No sensitive information detected."
                    )

    except KeyboardInterrupt:

        print()
        print("Stopping SafeSight...")

    finally:

        # =====================================================
        # ALWAYS REMOVE THE OVERLAY
        # =====================================================
        try:
            overlay.clear()
        except Exception:
            pass

        try:
            overlay.close()
        except Exception:
            pass

        print("Overlay removed.")
        print("SafeSight stopped.")


if __name__ == "__main__":
    main()