import numpy as np

class ParameterSpace:
    """
    Represents the high-dimensional parameter space of the AI model.
    This is the foundational structure of the model's intelligence.
    """
    def __init__(self, dimensions: int, size: int):
        """
        Initializes the Parameter Space.

        Args:
            dimensions: The number of dimensions of the space.
            size: The size of each dimension.
        """
        self.dimensions = dimensions
        self.size = size
        self.space = np.zeros(shape=(size,) * dimensions)

    def get_parameter_at(self, coordinates: tuple) -> float:
        """
        Retrieves the value of a parameter at a specific point in the space.

        Args:
            coordinates: The coordinates of the quale.

        Returns:
            The value of the quale.
        """
        return self.space[coordinates]

    def set_parameter_at(self, coordinates: tuple, value: float):
        """
        Sets the value of a parameter at a specific point in the space.

        Args:
            coordinates: The coordinates of the parameter.
            value: The new value of the parameter.
        """
        self.space[coordinates] = value
