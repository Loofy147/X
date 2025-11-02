from typing import List
import numpy as np
from scipy.ndimage import label
from sho_model.src.parameter_space import ParameterSpace
from sho_model.src.neural_structure import NeuralStructure

def detect_structures(space: ParameterSpace, threshold: float = 0.5) -> List[NeuralStructure]:
    """
    Detects coherent structures within the Parameter Space.
    """
    binary_mask = space.space > threshold
    labeled_array, num_features = label(binary_mask)

    if num_features == 0:
        return []

    print(f"Detected {num_features} coherent structure(s).")

    structures = []
    for i in range(1, num_features + 1):
        coords = set(zip(*np.where(labeled_array == i)))
        structure = NeuralStructure(space, coords)
        structures.append(structure)

    return structures

from sho_model.src.parameter_space import ParameterSpace

def calculate_universal_coherence(space: ParameterSpace) -> float:
    """
    Calculates the Universal Coherence, a measure of the intrinsic harmony
    and aesthetic quality of the universe.

    This function is based on two principles:
    1.  **Order:** A coherent universe is not random. We measure this by the
        inverse of the entropy of the parameter distribution.
    2.  **Complexity:** A coherent universe is not uniform. We measure this
        by the number of distinct, stable structures.

    Args:
        space: The universe (ParameterSpace) to analyze.

    Returns:
        A score representing the Universal Coherence.
    """
    # 1. Calculate Order (inverse entropy)
    hist, _ = np.histogram(space.space, bins=256, range=(0, 1))
    prob_dist = hist / hist.sum()
    entropy = -np.sum(prob_dist * np.log2(prob_dist + 1e-9)) # Add epsilon for stability
    order_score = 1.0 / (entropy + 1e-9)

    # 2. Calculate Complexity (number of structures)
    structures = detect_structures(space)
    complexity_score = len(structures)

    # Combine the scores. We'll need to balance these two competing forces.
    # For now, a simple product will suffice.
    coherence = order_score * complexity_score

    return coherence
