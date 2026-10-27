import re
import cv2
import pytesseract

from pytesseract import Output


# ============================================================
# TESSERACT CONFIGURATION
# ============================================================

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# ============================================================
# SENSITIVE INFORMATION DETECTOR
# ============================================================

class SensitiveInfoDetector:

    def __init__(self):

        self.patterns = {

            "EMAIL": re.compile(
                r"\b[A-Za-z0-9._%+-]+"
                r"@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
            ),

            "PHONE": re.compile(
                r"(?<!\d)"
                r"(?:\+91[\s-]?)?"
                r"[6-9]\d{9}"
                r"(?!\d)"
            ),

            "CARD_NUMBER": re.compile(
                r"(?<!\d)"
                r"(?:\d{4}[\s-]?){3}\d{4}"
                r"(?!\d)"
            ),

            "API_KEY": re.compile(
                r"\b(?:sk-|pk-|api[_-]?key[_-]?)"
                r"[A-Za-z0-9_\-]{12,}\b",
                re.IGNORECASE
            ),

            "PASSWORD": re.compile(
                r"\b(?:password|passwd|pwd)"
                r"\s*[:=]\s*\S+",
                re.IGNORECASE
            ),
        }


    # ========================================================
    # OCR
    # ========================================================

    def extract_words(self, image):

        data = pytesseract.image_to_data(
            image,
            output_type=Output.DICT,
            config="--psm 6"
        )

        words = []

        for i, text in enumerate(data["text"]):

            text = text.strip()

            if not text:
                continue

            try:
                confidence = float(
                    data["conf"][i]
                )
            except (ValueError, TypeError):
                confidence = 0.0

            x = int(data["left"][i])
            y = int(data["top"][i])
            w = int(data["width"][i])
            h = int(data["height"][i])

            # Tesseract hierarchy
            block_num = int(
                data["block_num"][i]
            )

            par_num = int(
                data["par_num"][i]
            )

            line_num = int(
                data["line_num"][i]
            )

            words.append({
                "text": text,
                "confidence": confidence,
                "bbox": (x, y, w, h),
                "block_num": block_num,
                "par_num": par_num,
                "line_num": line_num
            })

        return words


    # ========================================================
    # GROUP WORDS BY OCR LINE
    # ========================================================

    def group_lines(self, words):

        lines = {}

        for word in words:

            key = (
                word["block_num"],
                word["par_num"],
                word["line_num"]
            )

            if key not in lines:
                lines[key] = []

            lines[key].append(word)


        # Sort words from left to right
        for key in lines:

            lines[key].sort(
                key=lambda item: item["bbox"][0]
            )


        return list(lines.values())


    # ========================================================
    # DETECT SENSITIVE INFORMATION
    # ========================================================

    def detect(self, image):

        words = self.extract_words(image)

        lines = self.group_lines(words)

        results = []


        # ====================================================
        # PROCESS EACH OCR LINE SEPARATELY
        # ====================================================

        for line_words in lines:

            if not line_words:
                continue


            # ------------------------------------------------
            # Build text for this line
            # ------------------------------------------------

            combined_text = " ".join(
                word["text"]
                for word in line_words
            )


            # Text without spaces is useful for card numbers
            normalized_text = re.sub(
                r"[\s-]+",
                "",
                combined_text
            )


            # ------------------------------------------------
            # Test every sensitive pattern
            # ------------------------------------------------

            for info_type, pattern in self.patterns.items():

                # ============================================
                # Normal text search
                # ============================================

                match = pattern.search(
                    combined_text
                )


                # ============================================
                # Card-number search on normalized text
                # ============================================

                if (
                    info_type == "CARD_NUMBER"
                    and not match
                ):

                    normalized_match = pattern.search(
                        normalized_text
                    )

                    if normalized_match:

                        match_text = (
                            normalized_match.group(0)
                        )

                        match = normalized_match

                        # Map normalized match back
                        # to the complete line.
                        selected_words = (
                            line_words
                        )

                    else:

                        continue

                else:

                    if not match:
                        continue

                    match_text = match.group(0)

                    # ----------------------------------------
                    # Find words contributing to the match
                    # ----------------------------------------

                    selected_words = self._words_for_match(
                        line_words,
                        match,
                        combined_text
                    )


                if not selected_words:
                    continue


                # ------------------------------------------------
                # Bounding box
                # ------------------------------------------------

                x1 = min(
                    word["bbox"][0]
                    for word in selected_words
                )

                y1 = min(
                    word["bbox"][1]
                    for word in selected_words
                )

                x2 = max(
                    word["bbox"][0]
                    + word["bbox"][2]
                    for word in selected_words
                )

                y2 = max(
                    word["bbox"][1]
                    + word["bbox"][3]
                    for word in selected_words
                )


                bbox = (
                    x1,
                    y1,
                    x2 - x1,
                    y2 - y1
                )


                # ------------------------------------------------
                # Confidence
                # ------------------------------------------------

                confidences = [
                    word["confidence"]
                    for word in selected_words
                    if word["confidence"] >= 0
                ]


                if confidences:

                    confidence = (
                        sum(confidences)
                        / len(confidences)
                    )

                else:

                    confidence = 0.0


                results.append({
                    "type": info_type,
                    "text": match_text,
                    "confidence": confidence,
                    "bbox": bbox
                })


        # ====================================================
        # REMOVE DUPLICATES
        # ====================================================

        return self._remove_duplicates(results)


    # ========================================================
    # FIND WORDS THAT BELONG TO A MATCH
    # ========================================================

    def _words_for_match(
        self,
        line_words,
        match,
        combined_text
    ):

        match_start = match.start()
        match_end = match.end()

        selected_words = []

        current_position = 0


        for word in line_words:

            word_text = word["text"]

            word_start = current_position

            word_end = (
                current_position
                + len(word_text)
            )


            # ------------------------------------------------
            # Does this word overlap the match?
            # ------------------------------------------------

            if (
                word_end > match_start
                and word_start < match_end
            ):

                selected_words.append(word)


            # Space between OCR words
            current_position = word_end + 1


        return selected_words


    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    def _remove_duplicates(self, results):

        unique = []

        seen = set()


        for result in results:

            key = (
                result["type"],
                result["text"],
                result["bbox"]
            )


            if key in seen:
                continue


            seen.add(key)

            unique.append(result)


        return unique


# ============================================================
# DRAW DETECTIONS
# ============================================================

def draw_sensitive_detections(
    image,
    detections
):

    output = image.copy()


    for detection in detections:

        x, y, w, h = detection["bbox"]

        info_type = detection["type"]

        confidence = detection["confidence"]


        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            2
        )


        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        label = (
            f"{info_type} "
            f"{confidence:.0f}%"
        )


        cv2.putText(
            output,
            label,
            (x, max(20, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2
        )


    return output