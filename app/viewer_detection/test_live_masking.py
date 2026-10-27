import time
import mss

from app.viewer_detection.live_screen_detector import (
    LiveScreenDetector
)

from app.viewer_detection.screen_mask_overlay import (
    ScreenMaskOverlay
)


# ============================================================
# SCREEN INFORMATION
# ============================================================

with mss.mss() as screen:

    monitor = screen.monitors[1]

    screen_width = monitor["width"]
    screen_height = monitor["height"]


# ============================================================
# CREATE DETECTOR
# ============================================================

detector = LiveScreenDetector()


# ============================================================
# CREATE OVERLAY
# ============================================================

overlay = ScreenMaskOverlay(
    screen_width,
    screen_height
)


# ============================================================
# START
# ============================================================

print()
print("=" * 70)
print("SafeSight - LIVE SENSITIVE INFORMATION MASKING")
print("=" * 70)
print()
print("The SafeSight preview window will NOT be shown.")
print("Your normal desktop remains usable.")
print()
print("Sensitive information will be masked directly on the screen.")
print()
print("Press Ctrl+C in the terminal to stop.")
print()


try:

    while True:

        # ----------------------------------------------------
        # Capture desktop
        # ----------------------------------------------------

        frame, detections = detector.detect()


        # ----------------------------------------------------
        # Update transparent overlay
        # ----------------------------------------------------

        overlay.update_masks(
            detections
        )


        overlay.show()


        # ----------------------------------------------------
        # Small delay
        # ----------------------------------------------------

        time.sleep(0.1)


except KeyboardInterrupt:

    print()
    print("Stopping SafeSight...")


finally:

    overlay.clear()

    overlay.close()

    detector.close()

    print("SafeSight masking stopped.")