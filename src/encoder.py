import numpy as np
from sentence_transformers import SentenceTransformer


class TextEncoder:
    """
    Converts text into vectors using a frozen pretrained model.

    This encoder acts as the coordinate system for our experiment.

    We will not retrain the SentenceTransformer model. Later, memories
    will modify how embeddings move through the latent space instead.
    """

    def __init__(self) -> None:
        """
        Load the pretrained sentence embedding model.

        all-MiniLM-L6-v2 converts text into 384-dimensional vectors.

        The first time this runs, SentenceTransformers may download
        the model files from Hugging Face.
        """

        self._model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    def encode(self, text: str) -> np.ndarray:
        """
        Convert one string into a normalized NumPy vector.

        Parameters
        ----------
        text:
            The text that should be represented in latent space.

        Returns
        -------
        np.ndarray
            A one-dimensional array with shape (384,).

        Example
        -------
        encoder = TextEncoder()
        vector = encoder.encode("Rufus is a cat.")
        """

        embedding = self._model.encode(
            text,

            # Normalize the result so its Euclidean length is 1.
            #
            # This makes cosine similarity simpler later because:
            #
            # cosine_similarity(a, b)
            # = dot(a, b) / (||a|| * ||b||)
            #
            # When both vector lengths equal 1:
            #
            # cosine_similarity(a, b) = dot(a, b)
            normalize_embeddings=True,
        )

        # SentenceTransformer already returns an array-like result,
        # but we explicitly convert it to a NumPy float32 array.
        #
        # float32 uses less memory than float64 and matches the common
        # numerical precision used by neural-network models.
        return np.asarray(
            embedding,
            dtype=np.float32,
        )