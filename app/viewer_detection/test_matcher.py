import numpy as np

from app.viewer_detection.matcher import (
    cosine_similarity,
    find_best_match,
)


def main():

    # Create a simple normalized embedding
    embedding = np.array(
        [1.0, 0.0, 0.0],
        dtype=np.float32,
    )

    same_embedding = np.array(
        [1.0, 0.0, 0.0],
        dtype=np.float32,
    )

    different_embedding = np.array(
        [0.0, 1.0, 0.0],
        dtype=np.float32,
    )

    print(
        "Same:",
        cosine_similarity(
            embedding,
            same_embedding,
        ),
    )

    print(
        "Different:",
        cosine_similarity(
            embedding,
            different_embedding,
        ),
    )

    enrolled = np.array(
        [
            same_embedding,
            different_embedding,
        ],
        dtype=np.float32,
    )

    result = find_best_match(
        embedding,
        enrolled,
        threshold=0.45,
    )

    print()
    print("Match result:")
    print(result)


if __name__ == "__main__":
    main()