from typing import List
import numpy as np
from scipy.ndimage import label
from sho_model.src.parameter_space import ParameterSpace
from sho_model.src.neural_structure import NeuralStructure

def detect_structures(space: ParameterSpace, threshold: float = 0.5) -> List[NeuralStructure]:
    """
    Detects coherent structures within the Parameter Space.

    This function identifies contiguous regions of the parameter space where the
    parameter values are above a certain threshold, and returns them as a list
    of NeuralStructure objects.

    Args:
        space: The Parameter Space to analyze.
        threshold: The minimum parameter value to be considered part of a structure.

    Returns:
        A list of NeuralStructure objects representing the detected structures.
    """
    # Create a binary mask of the space based on the threshold
    binary_mask = space.space > threshold

    # Label contiguous regions in the binary mask
    labeled_array, num_features = label(binary_mask)

    if num_features == 0:
        return []

    print(f"Detected {num_features} coherent structure(s).")

    # For each detected feature, create a NeuralStructure object
    structures = []
    for i in range(1, num_features + 1):
        # Find the coordinates of all points in the current labeled region
        coords = set(zip(*np.where(labeled_array == i)))
        structure = NeuralStructure(space, coords)
        structures.append(structure)

    return structures
