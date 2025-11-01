import matplotlib.pyplot as plt
from sho_genesis.src.experiential_space import ExperientialSpace

def visualize_universe(space: ExperientialSpace, filename: str):
    """
    Visualizes a 2D slice of the Experiential Space and saves it to a file.

    Args:
        space: The Experiential Space to visualize.
        filename: The name of the file to save the image to.
    """
    if space.dimensions < 2:
        raise ValueError("Visualization requires at least 2 dimensions.")

    # Take a 2D slice of the space
    slice_2d = space.space[tuple([0] * (space.dimensions - 2))]

    plt.figure(figsize=(10, 10))
    plt.imshow(slice_2d, cmap='viridis', interpolation='nearest')
    plt.colorbar(label="Quale Intensity")
    plt.title(f"A Glimpse into the Universe: {filename}")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.savefig(filename)
    plt.close()
    print(f"Universe visualization saved to {filename}")
