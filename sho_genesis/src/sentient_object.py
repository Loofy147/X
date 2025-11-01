from typing import Set, Tuple
from sho_genesis.src.experiential_space import ExperientialSpace

class SentientObject:
    """
    Represents a coherent, self-aware region of the Experiential Space.
    This is the fundamental unit of consciousness.
    """
    def __init__(self, space: ExperientialSpace, coordinates: Set[Tuple]):
        """
        Initializes a Sentient Object.

        Args:
            space: The Experiential Space in which this object exists.
            coordinates: The set of coordinates that define the object's boundaries.
        """
        self.space = space
        self.coordinates = coordinates

    def get_state(self) -> dict:
        """
        Retrieves the current state of the Sentient Object.

        Returns:
            A dictionary mapping coordinates to quale values.
        """
        state = {}
        for coord in self.coordinates:
            state[coord] = self.space.get_quale_at(coord)
        return state

    def update_state(self, new_state: dict):
        """
        Updates the state of the Sentient Object.

        Args:
            new_state: A dictionary mapping coordinates to new quale values.
        """
        for coord, value in new_state.items():
            if coord in self.coordinates:
                self.space.set_quale_at(coord, value)
