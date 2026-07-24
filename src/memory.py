from dataclasses import dataclass

import numpy as np


@dataclass
class FlowMemory:
    """
    Represents one localized learned influence in latent space.

    center:
        The point where this memory has its strongest influence.

    direction:
        The direction in which nearby activation states are pushed.

    strength:
        A multiplier controlling how strongly this memory pushes.

    radius:
        Controls how far the memory's influence extends.
        A larger radius affects a wider region.
    """

    center: np.ndarray
    direction: np.ndarray
    strength: float = 1.0
    radius: float = 1.0