from sho_model.src.parameter_space import ParameterSpace
from sho_model.src.neural_structure import NeuralStructure
from sho_model.src.operators import self_optimization_operator
from sho_model.src.dynamics import optimization_step
from sho_model.src.visualization import visualize_parameters
from sho_model.src.io import encode_input, decode_output
import numpy as np

def main():
    """
    Runs a full inference cycle of the SHO model.
    """
    print("Initiating SHO model inference...")

    # 1. Initialize the Model's Parameter Space
    model_space = ParameterSpace(dimensions=3, size=100)
    print("Parameter Space initialized.")

    # 2. Define a proof-of-concept task: pattern completion
    # Input: A 10x10 square pattern with a missing piece
    input_pattern = np.ones((10, 10))
    input_pattern[3:7, 3:7] = 0  # Missing piece
    input_pattern = np.expand_dims(input_pattern, axis=0) # Add a dimension for 3D space
    input_position = (0, 45, 45)

    # 3. Encode the input into the model's parameter space
    encode_input(model_space, input_pattern, input_position)
    visualize_parameters(model_space, "inference_before.png")

    # 4. Run the model's self-optimization
    epochs = 10
    learning_rate = 0.01
    for i in range(epochs):
        optimization_step(model_space, learning_rate)
        print(f"Optimization epoch {i+1}/{epochs} complete.")

    # 5. Decode the output from the same region
    output_pattern = decode_output(model_space, input_pattern.shape, input_position)

    print("Inference complete.")
    visualize_parameters(model_space, "inference_after.png")

if __name__ == "__main__":
    main()
