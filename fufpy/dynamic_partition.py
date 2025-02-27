import numba as nb
import numpy as np


class DynamicPartition:
    """Dynamic partition of a set of integers `[0, ..., n-1]`.

    Parameters
    ----------
    n_elements : integer
        The data structure represents a partition of `[0, ..., n_elements-1]`,
        which is initially the partition with each element belonging to a different subset.
    """

    def __init__(self, n_elements: int):
        assert isinstance(n_elements, int), "n_elements must be a positive integer"
        assert n_elements > 0, "n_elements must be a positive integer"
        self._n_elements = n_elements
        self._attributes = dynamic_partition_create(n_elements)

    def representative(self, x: int) -> int:
        """Find the current representative for the subset containing `x`.

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
        return dynamic_partition_representative(self._attributes, x)

    def union(self, x: int, y: int) -> bool:
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
        return dynamic_partition_union(self._attributes, x, y)

    def subset(self, x: int) -> np.array:
        """Elements in the subset containing `x`.

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
        return dynamic_partition_subset(self._attributes, x)

    def parts(self) -> list[np.array]:
        """Parts of the partition.

        Returns
        -------
        subsets : list[np.array(dtype=int)]
            All disjoint subsets in the data structure.
        """
        return dynamic_partition_parts(self._attributes)

    def _assert_element_in_structure(self, x: int) -> None:
        assert isinstance(x, int), "x must be an integer"
        assert x >= 0, "x must be a positive integer"
        assert (
            x < self._n_elements
        ), "x must be smaller than the number of elements in the union-find structure"


def dynamic_partition_create(n_elements: int) -> np.array:
    """
    Create a dynamic partition with `n_elements` elements.

    Parameters
    ----------
    n_elements : integer
        The number of elements in the set.

    Returns
    -------
    dynamic_partition : np.array(shape=(3, n_elements), dtype=int)
        The dynamic partition data structure.
    """
    res = np.empty((3, n_elements), dtype=int)
    # Sizes.
    res[0, :] = np.full(n_elements, 1, dtype=int)
    # Parents.
    res[1, :] = np.arange(n_elements, dtype=int)
    # Siblings.
    res[2, :] = np.arange(n_elements, dtype=int)
    return res


@nb.njit
def dynamic_partition_representative(uf: np.array, x: int) -> int:
    """
    Find the current representative for the subset containig `x`.

    Parameters
    ----------
    uf : np.array(shape=(3, n_elements), dtype=int)
        The dynamic partition data structure.
    x : integer
        Element for which to find the representative.

    Returns
    -------
    representative : integer
        The representative of the subset containing `x`.
    """
    parents = uf[1]
    while x != parents[x]:
        parents[x] = parents[parents[x]]
        x = parents[x]
    return x


@nb.njit
def dynamic_partition_union(uf: np.array, x: int, y: int) -> bool:
    """
    Merge the subsets containing `x` and `y`.

    Parameters
    ----------
    uf : np.array(shape=(3, n_elements), dtype=int)
        The dynamic partition data structure.
    x, y : integers
        Elements to merge.

    Returns
    -------
    merged : bool
        True if `x` and `y` were in disjoint sets, False otherwise.
    """
    sizes = uf[0]
    parents = uf[1]
    siblings = uf[2]

    xr = dynamic_partition_representative(uf, x)
    yr = dynamic_partition_representative(uf, y)
    if xr == yr:
        return False

    if (sizes[xr], yr) < (sizes[yr], xr):
        xr, yr = yr, xr
    parents[yr] = xr
    sizes[xr] += sizes[yr]
    siblings[xr], siblings[yr] = siblings[yr], siblings[xr]
    return True


@nb.njit
def dynamic_partition_subset(uf: np.array, x: int) -> np.array:
    """
    Elements in the subset containing `x`.

    Parameters
    ----------
    uf : np.array(shape=(3, n_elements), dtype=int)
        The dynamic partition data structure.
    x : integer
        Element for which to find subset.

    Returns
    -------
    subset : np.array(dtype=int)
        All elements in the subset containing `x`.
    """
    siblings = uf[2]

    result = [x]
    next_sibling = siblings[x]
    while next_sibling != x:
        result.append(next_sibling)
        next_sibling = siblings[next_sibling]
    return np.array(result)


@nb.njit
def dynamic_partition_parts(uf: np.array) -> list[np.array]:
    """
    Parts of the partition.

    Parameters
    ----------
    uf : np.array(shape=(3, n_elements), dtype=int)
        The dynamic partition data structure.

    Returns
    -------
    subsets : list[np.array(dtype=int)]
        All disjoint subsets in the data structure.
    """
    result = []
    n_elements = uf.shape[1]
    visited = np.full(n_elements, False)
    for x in range(n_elements):
        xr = dynamic_partition_representative(uf, x)
        if not visited[xr]:
            visited[xr] = True
            x_set = dynamic_partition_subset(uf, x)
            result.append(x_set)
    return result
