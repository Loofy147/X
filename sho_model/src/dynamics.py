import numpy as np
from sho_model.src.parameter_space import ParameterSpace
from sho_model.src.cognition import detect_structures
from scipy.ndimage import laplace

def optimization_step(space: ParameterSpace, learning_rate: float, structure_preservation_factor: float = 0.5):
    """
    The Structure-Aware Optimization Step.

    This function evolves the model's parameters while preserving the integrity
    of detected structures. It applies a weaker optimization to stable structures
    and a stronger optimization to the surrounding, incoherent space.

    Args:
        space: The Parameter Space to evolve.
        learning_rate: The overall rate at which coherence is enforced.
        structure_preservation_factor: A value between 0 and 1 that determines
                                         how strongly existing structures are preserved.
    """
    # 1. Detect existing structures
    structures = detect_structures(space)

    # 2. Create a "structure mask" that is 1 inside structures and 0 outside
    structure_mask = np.zeros_like(space.space)
    for s in structures:
        for coord in s.coordinates:
            structure_mask[coord] = 1

    # 3. Calculate the global Laplacian (the driving force of coherence)
    laplacian = laplace(space.space)

    # 4. Create a "learning rate mask" that is weaker inside structures
    learning_rate_mask = learning_rate * (1 - structure_mask * structure_preservation_factor)

    # 5. Apply the masked update
    space.space += learning_rate_mask * laplacian
