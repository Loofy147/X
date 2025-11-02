from sho_model.src.parameter_space import ParameterSpace
from scipy.ndimage import laplace

def optimization_step(space: ParameterSpace, learning_rate: float):
    """
    The Optimization Step.

    This function governs the evolution of the model's parameters. It acts to
    maximize a global coherence metric, which we model here as minimizing the
    Laplacian of the parameter space (i.e., making it smoother and more ordered).

    Args:
        space: The Parameter Space to evolve.
        learning_rate: The rate at which coherence is enforced.
    """
    # Calculate the Laplacian of the parameter space
    laplacian = laplace(space.space)

    # Update the parameter space to minimize the Laplacian (move towards coherence)
    space.space += learning_rate * laplacian
