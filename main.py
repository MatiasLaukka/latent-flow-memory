import numpy as np

from src.encoder import TextEncoder


def cosine_similarity(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    """
    Calculate cosine similarity between two normalized vectors.

    In general, cosine similarity is:

        dot(a, b) / (||a|| * ||b||)

    Our encoder normalizes every embedding so both lengths are 1.
    Therefore this simplifies to:

        dot(a, b)

    A result near:
        1.0  means very similar direction
        0.0  means unrelated or perpendicular
       -1.0  means opposite direction
    """

    return float(np.dot(first, second))


def main() -> None:
    # Load the frozen embedding model.
    encoder = TextEncoder()

    # These are the concepts whose positions we want to compare.
    texts = [
        "Rufus",
        "cat",
        "dog",
        "Matu's cat",
        "Matu's cat's name",
        "Helsinki",
    ]

    # Create a dictionary where:
    #
    # key   = original text
    # value = its 384-dimensional embedding
    #
    # This dictionary comprehension is equivalent to writing:
    #
    # embeddings = {}
    #
    # for text in texts:
    #     embeddings[text] = encoder.encode(text)
    embeddings = {
        text: encoder.encode(text)
        for text in texts
    }

    # Use "Matu's cat's name" as the query location.
    #
    # We want to inspect which existing concepts are naturally
    # close to this query before adding any learned memory.
    query = embeddings["Matu's cat's name"]

    print("Similarity to query: Matu's cat's name")
    print("-" * 45)

    for text, embedding in embeddings.items():
        # Measure the angle-based similarity between the query
        # and the current candidate concept.
        similarity = cosine_similarity(
            query,
            embedding,
        )

        # <20 reserves 20 characters and left-aligns the text.
        # .4f displays the similarity with four decimal places.
        print(
            f"{text:<20} "
            f"{similarity:.4f}"
        )


# This prevents main() from running if this file is imported
# by another Python module.
#
# It only runs when we execute:
#
# python main.py
if __name__ == "__main__":
    main()