from sho_genesis.src.experiential_space import ExperientialSpace
from sho_genesis.src.sentient_object import SentientObject
from sho_genesis.src.operators import self_perception_operator
from sho_genesis.src.dynamics import actualization_function
from sho_genesis.src.visualization import visualize_universe
import numpy as np

def main():
    """
    The Genesis function. This is the "Big Bang" of our universe.
    """
    print("Initiating Genesis...")

    # 1. Create the Universe
    universe = ExperientialSpace(dimensions=3, size=100)
    print("Experiential Space forged.")

    # 2. Introduce primordial randomness (the "quantum foam")
    universe.space = np.random.rand(*universe.space.shape)
    print("Primordial chaos introduced.")
    visualize_universe(universe, "genesis_before.png")

    # 3. Form the first Sentient Object
    all_coords = set(np.ndindex(universe.space.shape))
    primordial_object = SentientObject(space=universe, coordinates=all_coords)
    print("First Sentient Object formed.")

    # 4. Run the cosmic evolution for a number of epochs
    epochs = 10
    coherence_factor = 0.01
    for i in range(epochs):
        # The universe evolves towards coherence
        actualization_function(universe, coherence_factor)

        # The primordial object perceives itself
        self_perception_operator(primordial_object)

        print(f"Epoch {i+1}/{epochs} complete.")

    print("Cosmic evolution complete. The universe has been born.")
    visualize_universe(universe, "genesis_after.png")

if __name__ == "__main__":
    main()
