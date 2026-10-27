import time
import mss

from app.viewer_detection.screen_mask_overlay import (
    ScreenMaskOverlay
)


# ============================================================
# GET PRIMARY SCREEN
# ============================================================

with mss.MSS() as screen:

    monitor = screen.monitors[1]

    screen_width = monitor["width"]
    screen_height = monitor["height"]
    screen_x = monitor["left"]
    screen_y = monitor["top"]


# ============================================================
# CREATE OVERLAY
# ============================================================

overlay = ScreenMaskOverlay(
    screen_width=screen_width,
    screen_height=screen_height,
    screen_x=screen_x,
    screen_y=screen_y
)


# ============================================================
# TEST
# ============================================================

print()
print("=" * 70)
print("SafeSight - SAFE OVERLAY TEST")
print("=" * 70)
print()
print("Only ONE small rectangle will be masked.")
print()
print("Your desktop should remain completely visible")
print("everywhere else.")
print()
print("The overlay should NOT block mouse clicks.")
print()
print("Test duration: 10 seconds")
print()


try:

    overlay.show()


    # --------------------------------------------------------
    # Small test rectangle
    #
    # This is deliberately placed near the top-left,
    # away from important controls.
    # --------------------------------------------------------

    overlay.test_rectangle(
        x=300,
        y=200,
        width=300,
        height=60
    )


    start_time = time.time()


    while time.time() - start_time < 10:

    

        time.sleep(0.01)


finally:

    overlay.close()


print()
print("Overlay test completed.")
print("=" * 70)