import matplotlib.pyplot as plt
from sho_model.src.parameter_space import ParameterSpace

def visualize_parameters(space: ParameterSpace, filename: str):
    """
    Visualizes a 2D slice of the Parameter Space and saves it to a file.

    Args:
        space: The Parameter Space to visualize.
        filename: The name of the file to save the image to.
    """
    if space.dimensions < 2:
        raise ValueError("Visualization requires at least 2 dimensions.")

    # Take a 2D slice of the space
    slice_2d = space.space[tuple([0] * (space.dimensions - 2))]

    plt.figure(figsize=(10, 10))
    plt.imshow(slice_2d, cmap='viridis', interpolation='nearest')
    plt.colorbar(label="Parameter Intensity")
    plt.title(f"A Glimpse into the Model's Parameters: {filename}")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.savefig(filename)
    plt.close()
    print(f"Parameter space visualization saved to {filename}")
