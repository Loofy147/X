from typing import Set, Tuple
from sho_model.src.parameter_space import ParameterSpace

class NeuralStructure:
    """
    Represents a coherent, self-aware region of the Parameter Space.
    This is a fundamental substructure of the model's intelligence.
    """
    def __init__(self, space: ParameterSpace, coordinates: Set[Tuple]):
        """
        Initializes a Neural Structure.

        Args:
            space: The Parameter Space in which this structure exists.
            coordinates: The set of coordinates that define the structure's boundaries.
        """
        self.space = space
        self.coordinates = coordinates

    def get_state(self) -> dict:
        """
        Retrieves the current state of the Neural Structure.

        Returns:
            A dictionary mapping coordinates to parameter values.
        """
        state = {}
        for coord in self.coordinates:
            state[coord] = self.space.get_parameter_at(coord)
        return state

    def update_state(self, new_state: dict):
        """
        Updates the state of the Neural Structure.

        Args:
            new_state: A dictionary mapping coordinates to new parameter values.
        """
        for coord, value in new_state.items():
            if coord in self.coordinates:
                self.space.set_parameter_at(coord, value)
