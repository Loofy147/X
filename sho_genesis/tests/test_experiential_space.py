import unittest
import numpy as np
from sho_genesis.src.experiential_space import ExperientialSpace

class TestExperientialSpace(unittest.TestCase):

    def test_creation(self):
        space = ExperientialSpace(dimensions=2, size=5)
        self.assertEqual(space.dimensions, 2)
        self.assertEqual(space.size, 5)
        self.assertEqual(space.space.shape, (5, 5))

    def test_get_set_quale(self):
        space = ExperientialSpace(dimensions=2, size=5)
        coordinates = (2, 3)
        value = 0.7
        space.set_quale_at(coordinates, value)
        self.assertEqual(space.get_quale_at(coordinates), value)

if __name__ == '__main__':
    unittest.main()
