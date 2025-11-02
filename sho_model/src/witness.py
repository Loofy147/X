from sho_model.src.parameter_space import ParameterSpace
from sho_model.src.dynamics import optimization_step
from sho_model.src.visualization import visualize_parameters
from sho_model.src.cognition import calculate_universal_coherence
import numpy as np

def main():
    """
    Witnesses the self-creation of the SHO universe.
    """
    print("Initiating the Great Work...")

    # --- Hyperparameters ---
    generations = 10
    epochs_per_generation = 5

    # --- Dynamic Laws of the Universe ---
    learning_rate = 0.01
    structure_preservation_factor = 0.5

    # --- The Primordial State ---
    universe = ParameterSpace(dimensions=3, size=100)
    universe.space = np.random.rand(*universe.space.shape) # Start with chaos

    for gen in range(generations):
        print(f"\n--- Generation {gen+1}/{generations} ---")
        print(f"Current Laws: LR={learning_rate:.4f}, SPF={structure_preservation_factor:.4f}")

        # 1. Evolve the universe for a number of epochs
        for i in range(epochs_per_generation):
            optimization_step(universe, learning_rate, structure_preservation_factor)

        # 2. Measure the universe's internal coherence
        coherence = calculate_universal_coherence(universe)
        print(f"Generation {gen+1} Universal Coherence: {coherence:.4f}")

        # 3. Meta-Evolution: Adjust the laws to maximize coherence
        # This is the universe's own will to become more beautiful.
        # We'll use a simple hill-climbing approach for now.
        if gen > 0 and coherence > last_coherence:
            # If we are improving, amplify the current direction of change.
            structure_preservation_factor = min(0.99, structure_preservation_factor + 0.01)
        else:
            # If we are not improving, explore a different path.
            structure_preservation_factor = max(0.01, structure_preservation_factor - 0.02)

        last_coherence = coherence

        # 4. Visualize the state of the universe
        if (gen + 1) % 10 == 0:
            visualize_parameters(universe, f"universe_gen_{gen+1}.png")

    print("\nThe Great Work is complete. The universe has found a state of high coherence.")
    visualize_parameters(universe, "universe_final_state.png")

if __name__ == "__main__":
    main()
