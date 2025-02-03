import numba as nb
import numpy as np


class UnionFind:
    def __init__(self, n_elements):
        """Union-find (aka disjoint-set) on a set of integers `[0, ..., n-1]`.

        Parameters
        ----------
        n_elements : integer
            The data structure represents a partition of `[0, ..., n_elements-1]`,
            which is initially the partition with each element belonging to a different subset.
        """
        assert type(n_elements) == int, "n_elements must be a positive integer"
        assert n_elements > 0, "n_elements must be a positive integer"
        self._n_elements = n_elements
        self._attributes = unionfind_create(n_elements)

    def find(self, x):
        """Find the current representative for the subset of `x`.

        Parameters
        ----------
        x : integer
            Element for which to find a representative.

        Returns
        -------
        representative : integer
            The element representing the subset of `x`.


        """
        self._assert_element_in_structure(x)
        return unionfind_find(self._attributes, x)

    def union(self, x, y):
        """Merge the subsets of `x` and `y`.

        Parameters
        ----------
        x, y : integers
            Elements to merge.

        Returns
        -------
        merged : bool
            True if `x` and `y` were in disjoint sets, False otherwise.
        """
        self._assert_element_in_structure(x)
        self._assert_element_in_structure(y)
        return unionfind_union(self._attributes, x, y)

    def subset(self, x):
        """All elements in the subset of `x`.

        Parameters
        ----------
        x : integer
            Element for which to find subset.

        Returns
        -------
        subset : np.array(dtype=int)
            All elements in the subset of `x`.
        """
        self._assert_element_in_structure(x)
        return unionfind_subset(self._attributes, x)

    def subsets(self):
        """All subsets.

        Returns
        -------
        subsets : list[np.array(dtype=int)]
            All disjoint subsets in the data structure.
        """
        return unionfind_subsets(self._attributes)

    def _assert_element_in_structure(self, x):
        assert type(x) == int, "x must be an integer"
        assert x >= 0, "x must be a positive integer"
        assert (
            x < self._n_elements
        ), "x must be smaller than the number of elements in the union-find structure"


def unionfind_create(n_elements):
    res = np.empty((4, n_elements), dtype=int)
    # sizes
    res[0, :] = np.full(n_elements, 1, dtype=int)
    # parents
    res[1, :] = np.arange(n_elements, dtype=int)
    # neighbors
    res[2, :] = np.arange(n_elements, dtype=int)
    return res


@nb.njit
def unionfind_find(uf, x):
    parents = uf[1]
    while x != parents[x]:
        parents[x] = parents[parents[x]]
        x = parents[x]
    return x


@nb.njit
def unionfind_union(uf, x, y):
    sizes = uf[0]
    parents = uf[1]
    neighbors = uf[2]

    xr = unionfind_find(uf, x)
    yr = unionfind_find(uf, y)
    if xr == yr:
        return False

    if (sizes[xr], yr) < (sizes[yr], xr):
        xr, yr = yr, xr
    parents[yr] = xr
    sizes[xr] += sizes[yr]
    neighbors[xr], neighbors[yr] = neighbors[yr], neighbors[xr]
    return True


@nb.njit
def unionfind_subset(uf, x):
    neighbors = uf[2]

    result = [x]
    nxt = neighbors[x]
    while nxt != x:
        result.append(nxt)
        nxt = neighbors[nxt]
    return np.array(result)


@nb.njit
def unionfind_subsets(uf):
    result = []
    visited = set()
    for x in range(uf.shape[1]):
        if x not in visited:
            xset = unionfind_subset(uf, x)
            visited.update(xset)
            result.append(xset)
    return result
