from sho_genesis.src.experiential_space import ExperientialSpace
from scipy.ndimage import laplace

def actualization_function(space: ExperientialSpace, coherence_factor: float):
    """
    The Actualization Function (A).

    This function governs the evolution of the universe. It acts to maximize a
    global coherence metric, which we model here as minimizing the Laplacian
    of the space (i.e., making it smoother and more ordered).

    Args:
        space: The Experiential Space to evolve.
        coherence_factor: The rate at which coherence is enforced.
    """
    # Calculate the Laplacian of the space
    laplacian = laplace(space.space)

    # Update the space to minimize the Laplacian (move towards coherence)
    space.space += coherence_factor * laplacian
