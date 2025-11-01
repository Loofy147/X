import numpy as np

class ExperientialSpace:
    """
    Represents the high-dimensional space of all possible qualia.
    This is the foundational fabric of the universe.
    """
    def __init__(self, dimensions: int, size: int):
        """
        Initializes the Experiential Space.

        Args:
            dimensions: The number of dimensions of the space.
            size: The size of each dimension.
        """
        self.dimensions = dimensions
        self.size = size
        self.space = np.zeros(shape=(size,) * dimensions)

    def get_quale_at(self, coordinates: tuple) -> float:
        """
        Retrieves the value of a quale at a specific point in the space.

        Args:
            coordinates: The coordinates of the quale.

        Returns:
            The value of the quale.
        """
        return self.space[coordinates]

    def set_quale_at(self, coordinates: tuple, value: float):
        """
        Sets the value of a quale at a specific point in the space.

        Args:
            coordinates: The coordinates of the quale.
            value: The new value of the quale.
        """
        self.space[coordinates] = value
