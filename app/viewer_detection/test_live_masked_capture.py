import time
import cv2
import mss
import numpy as np

from app.viewer_detection.sensitive_info_detector import (
    SensitiveInfoDetector
)

from app.viewer_detection.screen_masker import (
    ScreenMasker
)


# ============================================================
# SCREEN CAPTURE
# ============================================================

screen = mss.MSS()

monitor = screen.monitors[1]

detector = SensitiveInfoDetector()

masker = ScreenMasker(
    padding=5
)


MASKED_OUTPUT = "data/live_masked_screen.png"
RAW_OUTPUT = "data/live_raw_screen.png"


# ============================================================
# START
# ============================================================

print()
print("=" * 70)
print("SafeSight - LIVE SCREEN MASKING TEST")
print("=" * 70)
print()
print("Your real desktop will NOT be modified.")
print()
print("A masked COPY will be saved whenever sensitive")
print("information is detected.")
print()
print("Press Ctrl+C to stop.")
print()


last_state = None

try:

    while True:

        # ----------------------------------------------------
        # Capture current desktop
        # ----------------------------------------------------

        screenshot = screen.grab(
            monitor
        )

        frame = np.array(
            screenshot
        )

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGRA2BGR
        )


        # ----------------------------------------------------
        # Detect sensitive information
        # ----------------------------------------------------

        detections = detector.detect(
            frame
        )


        # ----------------------------------------------------
        # Apply masks
        # ----------------------------------------------------

        masked_frame = masker.apply_masks(
            frame,
            detections
        )


        # ----------------------------------------------------
        # Save RAW screenshot
        # ----------------------------------------------------

        cv2.imwrite(
            RAW_OUTPUT,
            frame
        )


        # ----------------------------------------------------
        # IMPORTANT:
        # Save masked image ONLY when detections exist.
        #
        # Therefore a later frame with 0 detections
        # cannot overwrite our last valid masked image.
        # ----------------------------------------------------

        if detections:

            cv2.imwrite(
                MASKED_OUTPUT,
                masked_frame
            )


        # ----------------------------------------------------
        # Detect state changes
        # ----------------------------------------------------

        current_state = tuple(
            (
                detection["type"],
                detection["text"],
                detection["bbox"]
            )
            for detection in detections
        )


        if current_state != last_state:

            print()

            print(
                f"Sensitive items: "
                f"{len(detections)}"
            )


            if detections:

                for detection in detections:

                    print(
                        f"  {detection['type']:12} "
                        f"-> MASKED"
                    )

                print(
                    f"\nMasked image saved to:"
                )

                print(
                    f"  {MASKED_OUTPUT}"
                )

            else:

                print(
                    "No sensitive information detected."
                )


            last_state = current_state


        time.sleep(0.5)


except KeyboardInterrupt:

    print()
    print("Stopping SafeSight...")


finally:

    screen.close()

    print()
    print(
        "Last masked screen remains saved at:"
    )

    print(
        MASKED_OUTPUT
    )

    print()
    print(
        "Raw screen saved at:"
    )

    print(
        RAW_OUTPUT
    )