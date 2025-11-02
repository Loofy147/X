from sho_model.src.parameter_space import ParameterSpace
from sho_model.src.dynamics import optimization_step
from sho_model.src.visualization import visualize_parameters
from sho_model.src.io import encode_input, decode_output
from sho_model.src.cognition import calculate_reward
import numpy as np

def main():
    """
    Runs a meta-learning training loop for the SHO model.
    """
    print("Initiating SHO model training loop...")

    # --- Hyperparameters ---
    training_generations = 10
    epochs_per_generation = 10

    # --- Dynamic Laws of Physics (Meta-parameters) ---
    learning_rate = 0.01
    structure_preservation_factor = 0.5

    # --- Define the Task ---
    input_pattern = np.ones((10, 10))
    input_pattern[3:7, 3:7] = 0
    input_pattern = np.expand_dims(input_pattern, axis=0)
    input_position = (0, 45, 45)
    ideal_output = np.ones_like(input_pattern)

    for gen in range(training_generations):
        print(f"\n--- Generation {gen+1}/{training_generations} ---")
        print(f"Current Laws: LR={learning_rate:.4f}, SPF={structure_preservation_factor:.4f}")

        # 1. Initialize the model for this generation
        model_space = ParameterSpace(dimensions=3, size=100)
        encode_input(model_space, input_pattern, input_position)

        # 2. Run the optimization process
        for i in range(epochs_per_generation):
            optimization_step(model_space, learning_rate, structure_preservation_factor)

        # 3. Get the result and calculate the reward
        output = decode_output(model_space, input_pattern.shape, input_position)
        reward = calculate_reward(output, ideal_output)
        print(f"Generation {gen+1} Reward: {reward:.4f}")

        # 4. Meta-Learning: Adjust the laws based on the reward
        # Simple strategy: If reward is low, increase structure preservation to be more careful.
        if reward < 0.95:
            structure_preservation_factor = min(0.99, structure_preservation_factor + 0.05)
        else:
            structure_preservation_factor = max(0.01, structure_preservation_factor - 0.05)

        # Visualize the final state of this generation
        visualize_parameters(model_space, f"training_gen_{gen+1}.png")

    print("\nTraining complete.")

if __name__ == "__main__":
    main()
