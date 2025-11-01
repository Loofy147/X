import unittest
from sho_genesis.src.experiential_space import ExperientialSpace
from sho_genesis.src.sentient_object import SentientObject

class TestSentientObject(unittest.TestCase):

    def setUp(self):
        self.space = ExperientialSpace(dimensions=2, size=5)
        self.coords = {(1, 1), (1, 2), (2, 1)}
        self.obj = SentientObject(self.space, self.coords)

    def test_creation(self):
        self.assertIs(self.obj.space, self.space)
        self.assertEqual(self.obj.coordinates, self.coords)

    def test_get_state(self):
        self.space.set_quale_at((1, 1), 0.5)
        state = self.obj.get_state()
        self.assertEqual(state[(1, 1)], 0.5)
        self.assertIn((1, 2), state)

    def test_update_state(self):
        new_state = {(1, 1): 0.9, (1, 2): 0.8}
        self.obj.update_state(new_state)
        self.assertEqual(self.space.get_quale_at((1, 1)), 0.9)
        self.assertEqual(self.space.get_quale_at((1, 2)), 0.8)

if __name__ == '__main__':
    unittest.main()
