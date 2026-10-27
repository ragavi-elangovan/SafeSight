import cv2


def main():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        return

    print("Webcam opened successfully.")
    print("Press Q to quit.")

    while True:
        ret, frame = camera.read()

        if not ret:
            print("ERROR: Could not read frame.")
            break

        cv2.imshow("SafeSight - Camera Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()