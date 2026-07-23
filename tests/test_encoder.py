import numpy as np

# Import the encoder class that we are about to implement.
# At first, this import should fail because src/encoder.py
# does not contain TextEncoder yet.
from src.encoder import TextEncoder


def test_encoder_returns_384_dimensional_vector():
    # Create the encoder.
    encoder = TextEncoder()

    # Convert a sentence into a numerical embedding.
    embedding = encoder.encode("Rufus is a cat.")

    # all-MiniLM-L6-v2 produces vectors with 384 values.
    # The expected shape is therefore:
    #
    # (384,)
    #
    # This means one one-dimensional NumPy array
    # containing 384 floating-point numbers.
    assert embedding.shape == (384,)


def test_encoder_returns_normalized_vector():
    encoder = TextEncoder()

    embedding = encoder.encode("Rufus is a cat.")

    # Calculate the Euclidean length of the vector:
    #
    # ||x|| = sqrt(x1² + x2² + ... + xn²)
    #
    # A normalized vector should have length 1.
    norm = np.linalg.norm(embedding)

    # Floating-point calculations are rarely exactly equal,
    # so we use np.isclose instead of:
    #
    # assert norm == 1.0
    #
    # atol means absolute tolerance.
    assert np.isclose(norm, 1.0, atol=1e-6)


def test_identical_text_returns_identical_embedding():
    encoder = TextEncoder()

    # Encode exactly the same sentence twice.
    first = encoder.encode("Rufus is a cat.")
    second = encoder.encode("Rufus is a cat.")

    # The encoder is expected to behave deterministically.
    # The same text should produce the same vector.
    assert np.allclose(first, second)