from insightface.app import FaceAnalysis


def main():
    print("Loading InsightFace...")

    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"],
    )

    app.prepare(
        ctx_id=0,
        det_size=(640, 640),
    )

    print("InsightFace loaded successfully.")
    print("ArcFace recognition model is ready.")


if __name__ == "__main__":
    main()