import cv2


class ScreenMasker:

    def __init__(
        self,
        padding=4,
        mask_color=(0, 0, 0)
    ):
        self.padding = padding
        self.mask_color = mask_color


    # ========================================================
    # MASK ONE REGION
    # ========================================================

    def mask_region(
        self,
        image,
        bbox
    ):

        x, y, w, h = bbox

        height, width = image.shape[:2]

        x1 = max(
            0,
            x - self.padding
        )

        y1 = max(
            0,
            y - self.padding
        )

        x2 = min(
            width,
            x + w + self.padding
        )

        y2 = min(
            height,
            y + h + self.padding
        )

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            self.mask_color,
            -1
        )

        return image


    # ========================================================
    # MASK ALL SENSITIVE INFORMATION
    # ========================================================

    def apply_masks(
        self,
        image,
        detections
    ):

        masked_image = image.copy()

        for detection in detections:

            self.mask_region(
                masked_image,
                detection["bbox"]
            )

        return masked_image