import unittest
from fufpy import UnionFind


class TestUnionFind(unittest.TestCase):
    def test_init(self):
        """Initialize a one-element union-find,
        and check that the representative of `0` is `0`."""
        n_elements = 1
        uf = UnionFind(n_elements)
        assert uf.find(0) == 0


if __name__ == "__main__":
    unittest.main()
