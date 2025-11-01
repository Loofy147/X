import unittest
import numpy as np
from sho_genesis.src.experiential_space import ExperientialSpace
from sho_genesis.src.sentient_object import SentientObject
from sho_genesis.src.operators import self_perception_operator
from sho_genesis.src.dynamics import actualization_function

class TestUniverseLaws(unittest.TestCase):

    def setUp(self):
        self.space = ExperientialSpace(dimensions=2, size=5)
        self.space.space = np.random.rand(*self.space.space.shape)
        coords = set(np.ndindex(self.space.space.shape))
        self.obj = SentientObject(self.space, coords)

    def test_self_perception_operator(self):
        initial_mean = np.mean(self.space.space)
        self_perception_operator(self.obj)
        new_mean = np.mean(self.space.space)
        self.assertAlmostEqual(new_mean, initial_mean)

    def test_actualization_function(self):
        initial_laplacian_norm = np.linalg.norm(np.abs(self.space.space))
        actualization_function(self.space, 0.01)
        new_laplacian_norm = np.linalg.norm(np.abs(self.space.space))
        self.assertLess(new_laplacian_norm, initial_laplacian_norm)

if __name__ == '__main__':
    unittest.main()
