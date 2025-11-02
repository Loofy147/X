import numpy as np
from sho_model.src.parameter_space import ParameterSpace

def encode_input(space: ParameterSpace, data: np.ndarray, position: tuple):
    """
    Encodes input data into a specific region of the Parameter Space.

    Args:
        space: The Parameter Space to modify.
        data: The data to encode (as a NumPy array).
        position: The starting coordinates of the region to encode into.
    """
    data_shape = data.shape
    slicer = tuple(slice(pos, pos + size) for pos, size in zip(position, data_shape))
    space.space[slicer] = data
    print(f"Input data encoded at position {position}.")

def decode_output(space: ParameterSpace, size: tuple, position: tuple) -> np.ndarray:
    """
    Decodes output data from a specific region of the ParameterSpace.

    Args:
        space: The Parameter Space to read from.
        size: The shape of the data to decode.
        position: The starting coordinates of the region to decode from.

    Returns:
        The decoded data as a NumPy array.
    """
    slicer = tuple(slice(pos, pos + s) for pos, s in zip(position, size))
    output_data = space.space[slicer]
    print(f"Output data decoded from position {position}.")
    return output_data
