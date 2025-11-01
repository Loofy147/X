import unittest
import numpy as np
from sho_model.src.parameter_space import ParameterSpace

class TestParameterSpace(unittest.TestCase):

    def test_creation(self):
        space = ParameterSpace(dimensions=2, size=5)
        self.assertEqual(space.dimensions, 2)
        self.assertEqual(space.size, 5)
        self.assertEqual(space.space.shape, (5, 5))

    def test_get_set_parameter(self):
        space = ParameterSpace(dimensions=2, size=5)
        coordinates = (2, 3)
        value = 0.7
        space.set_parameter_at(coordinates, value)
        self.assertEqual(space.get_parameter_at(coordinates), value)

if __name__ == '__main__':
    unittest.main()
