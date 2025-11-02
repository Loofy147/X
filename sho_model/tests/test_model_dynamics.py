import unittest
import numpy as np
from sho_model.src.parameter_space import ParameterSpace
from sho_model.src.neural_structure import NeuralStructure
from sho_model.src.operators import self_optimization_operator
from sho_model.src.dynamics import optimization_step

class TestModelDynamics(unittest.TestCase):

    def setUp(self):
        self.space = ParameterSpace(dimensions=2, size=5)
        self.space.space = np.random.rand(*self.space.space.shape)
        coords = set(np.ndindex(self.space.space.shape))
        self.structure = NeuralStructure(self.space, coords)

    def test_self_optimization_operator(self):
        initial_mean = np.mean(self.space.space)
        self_optimization_operator(self.structure)
        new_mean = np.mean(self.space.space)
        self.assertAlmostEqual(new_mean, initial_mean)

    def test_optimization_step(self):
        # The optimization step should decrease the standard deviation, as it smooths the space
        initial_std = np.std(self.space.space)
        optimization_step(self.space, 0.01)
        new_std = np.std(self.space.space)
        self.assertLess(new_std, initial_std)

if __name__ == '__main__':
    unittest.main()
