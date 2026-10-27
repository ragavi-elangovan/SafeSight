import time
import mss
import numpy as np
import cv2

from app.viewer_detection.sensitive_info_detector import (
    SensitiveInfoDetector
)

from app.viewer_detection.screen_mask_overlay import (
    ScreenMaskOverlay
)


# ============================================================
# SCREEN CONFIGURATION
# ============================================================

screen = mss.MSS()

monitor = screen.monitors[1]

screen_width = monitor["width"]
screen_height = monitor["height"]

screen_x = monitor["left"]
screen_y = monitor["top"]


# ============================================================
# CREATE SENSITIVE INFORMATION DETECTOR
# ============================================================

detector = SensitiveInfoDetector()


# ============================================================
# CREATE WINDOWS OVERLAY
# ============================================================

overlay = ScreenMaskOverlay(
    screen_width=screen_width,
    screen_height=screen_height,
    screen_x=screen_x,
    screen_y=screen_y
)


# ============================================================
# OCR INTERVAL
# ============================================================

DETECTION_INTERVAL = 0.5

last_detection_time = 0.0

last_state = None


# ============================================================
# START
# ============================================================

print()
print("=" * 70)
print("SafeSight - LIVE SENSITIVE INFORMATION OVERLAY")
print("=" * 70)
print()
print("SafeSight is now monitoring your screen.")
print()
print("Sensitive information will be masked DIRECTLY")
print("on the application you are currently using.")
print()
print("No screenshot preview window will be shown.")
print("No PNG file will be created.")
print()
print("Press Ctrl+C in this terminal to stop.")
print()
print("=" * 70)
print()


try:

    # --------------------------------------------------------
    # Show transparent overlay
    # --------------------------------------------------------

    overlay.show()


    while True:

        current_time = time.time()


        # ----------------------------------------------------
        # Capture screen only at OCR interval
        # ----------------------------------------------------

        if (
            current_time - last_detection_time
            >= DETECTION_INTERVAL
        ):

            screenshot = screen.grab(
                monitor
            )


            # ------------------------------------------------
            # Convert MSS screenshot to OpenCV format
            # ------------------------------------------------

            frame = np.array(
                screenshot
            )

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGRA2BGR
            )


            # ------------------------------------------------
            # Detect sensitive information
            # ------------------------------------------------

            detections = detector.detect(
                frame
            )


            last_detection_time = current_time


            # ------------------------------------------------
            # Update overlay
            # ------------------------------------------------

            overlay.update_masks(
                detections
            )


            # ------------------------------------------------
            # Print only when detection changes
            # ------------------------------------------------

            state = tuple(
                (
                    detection["type"],
                    detection["text"],
                    detection["bbox"]
                )
                for detection in detections
            )


            if state != last_state:

                print()
                print(
                    f"Sensitive items: "
                    f"{len(detections)}"
                )


                for detection in detections:

                    print(
                        f"  {detection['type']:12} "
                        f"-> MASKED"
                    )


                if not detections:

                    print(
                        "  Screen clear."
                    )


                last_state = state


        # ----------------------------------------------------
        # Keep Windows overlay responsive
        # ----------------------------------------------------

        overlay.root.update()

        time.sleep(0.01)


except KeyboardInterrupt:

    print()
    print("Stopping SafeSight...")


finally:

    # --------------------------------------------------------
    # Remove overlay BEFORE exiting
    # --------------------------------------------------------

    overlay.clear()

    overlay.close()

    screen.close()


    print()
    print("=" * 70)
    print("SafeSight live overlay stopped.")
    print("=" * 70)