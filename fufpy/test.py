import unittest
from fufpy import UnionFind


class TestUnionFind(unittest.TestCase):

    def test_init(self):
        """Initialize a one-element union-find,
        and check that the representative of `0` is `0`."""
        n_elements = 1
        uf = UnionFind(n_elements)
        assert uf.find(0) == 0

    def test_union_and_find(self):
        """Check that union in results in elements
        with the same representative."""

        n_elements = 10
        uf = UnionFind(n_elements)

        assert uf.find(0) != uf.find(1)
        uf.union(0,1)
        assert uf.find(0) == uf.find(1)

        assert uf.find(0) != uf.find(2)
        uf.union(1,2)
        assert uf.find(0) == uf.find(2)

        assert uf.find(1) != uf.find(3)
        uf.union(0,3)
        assert uf.find(1) == uf.find(3)

        assert uf.find(8) != uf.find(9)
        uf.union(8,9)
        assert uf.find(8) == uf.find(9)

        assert uf.find(8) != uf.find(7)
        uf.union(9,7)
        assert uf.find(8) == uf.find(7)

        assert uf.find(2) != uf.find(9)
        uf.union(8,0)
        assert uf.find(2) == uf.find(9)

    def test_subset(self):
        """Check that the subset corresponding to an element is correct."""

        n_elements = 10
        uf = UnionFind(n_elements)

        uf.union(0,1)
        uf.union(1,2)
        uf.union(0,3)
        uf.union(8,9)
        uf.union(9,7)
        uf.union(8,0)

        assert set(uf.subset(0)) == {0,1,2,3,7,8,9}


if __name__ == "__main__":
    unittest.main()
