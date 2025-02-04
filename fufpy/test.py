import unittest
from fufpy import DynamicPartition


class TestUnionFind(unittest.TestCase):

    def test_init(self):
        """Initialize a one-element union-find,
        and check that the representative of `0` is `0`."""
        n_elements = 1
        uf = DynamicPartition(n_elements)
        assert uf.representative(0) == 0

    def test_union_and_find(self):
        """Check that union in results in elements
        with the same representative."""

        n_elements = 10
        uf = DynamicPartition(n_elements)

        assert uf.representative(0) != uf.representative(1)
        uf.union(0, 1)
        assert uf.representative(0) == uf.representative(1)

        assert uf.representative(0) != uf.representative(2)
        uf.union(1, 2)
        assert uf.representative(0) == uf.representative(2)

        assert uf.representative(1) != uf.representative(3)
        uf.union(0, 3)
        assert uf.representative(1) == uf.representative(3)

        assert uf.representative(8) != uf.representative(9)
        uf.union(8, 9)
        assert uf.representative(8) == uf.representative(9)

        assert uf.representative(8) != uf.representative(7)
        uf.union(9, 7)
        assert uf.representative(8) == uf.representative(7)

        assert uf.representative(2) != uf.representative(9)
        uf.union(8, 0)
        assert uf.representative(2) == uf.representative(9)

    def test_subset(self):
        """Check that the subset corresponding to an element is correct."""

        n_elements = 10
        uf = DynamicPartition(n_elements)

        uf.union(0, 1)
        uf.union(1, 2)
        uf.union(0, 3)
        uf.union(8, 9)
        uf.union(9, 7)
        uf.union(8, 0)

        assert set(uf.subset(0)) == {0, 1, 2, 3, 7, 8, 9}

    def test_parts(self):
        """Check that the parts in the partition are correct."""

        n_elements = 10
        uf = DynamicPartition(n_elements)

        uf.union(0, 1)
        uf.union(1, 2)
        uf.union(0, 3)
        uf.union(8, 9)
        uf.union(9, 7)
        uf.union(8, 0)

        uf.union(4, 5)

        parts = uf.parts()
        assert len(parts) == 3

        expected_parts = [{6}, {4, 5}, {0, 1, 2, 3, 7, 8, 9}]

        for subset in parts:
            assert set(subset) in expected_parts


if __name__ == "__main__":
    unittest.main()
