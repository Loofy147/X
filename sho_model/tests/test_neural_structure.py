import unittest
from sho_model.src.parameter_space import ParameterSpace
from sho_model.src.neural_structure import NeuralStructure

class TestNeuralStructure(unittest.TestCase):

    def setUp(self):
        self.space = ParameterSpace(dimensions=2, size=5)
        self.coords = {(1, 1), (1, 2), (2, 1)}
        self.structure = NeuralStructure(self.space, self.coords)

    def test_creation(self):
        self.assertIs(self.structure.space, self.space)
        self.assertEqual(self.structure.coordinates, self.coords)

    def test_get_state(self):
        self.space.set_parameter_at((1, 1), 0.5)
        state = self.structure.get_state()
        self.assertEqual(state[(1, 1)], 0.5)
        self.assertIn((1, 2), state)

    def test_update_state(self):
        new_state = {(1, 1): 0.9, (1, 2): 0.8}
        self.structure.update_state(new_state)
        self.assertEqual(self.space.get_parameter_at((1, 1)), 0.9)
        self.assertEqual(self.space.get_parameter_at((1, 2)), 0.8)

if __name__ == '__main__':
    unittest.main()
