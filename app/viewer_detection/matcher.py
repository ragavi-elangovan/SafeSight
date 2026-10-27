import numpy as np


def cosine_similarity(
    embedding_a: np.ndarray,
    embedding_b: np.ndarray,
) -> float:

    embedding_a = np.asarray(
        embedding_a,
        dtype=np.float32,
    )

    embedding_b = np.asarray(
        embedding_b,
        dtype=np.float32,
    )

    norm_a = np.linalg.norm(embedding_a)
    norm_b = np.linalg.norm(embedding_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(
        np.dot(embedding_a, embedding_b)
        / (norm_a * norm_b)
    )


def find_best_match(
    query_embedding: np.ndarray,
    enrolled_embeddings: np.ndarray,
    threshold: float = 0.45,
):
    """
    Compare one live embedding against
    multiple enrolled embeddings.

    enrolled_embeddings shape:
        (N, 512)

    query_embedding shape:
        (512,)
    """

    best_score = -1.0
    best_index = -1

    for index, enrolled_embedding in enumerate(
        enrolled_embeddings
    ):

        score = cosine_similarity(
            query_embedding,
            enrolled_embedding,
        )

        if score > best_score:
            best_score = score
            best_index = index

    matched = best_score >= threshold

    return {
        "matched": matched,
        "index": best_index,
        "score": best_score,
    }