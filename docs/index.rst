FufPy
=====


Implementation of union-find (aka [disjoint-set](https://en.wikipedia.org/wiki/Disjoint-set_data_structure)) data structure.
Currently, for performance, the structure is defined on a set $\{0, \dots, n-1\}$, of size $n$, which is specified at initialization.

It implements the standard operations, as well as $subset$, which returns the subset corresponding to an element.
A main use case is [hierarchical clustering](https://en.wikipedia.org/wiki/Hierarchical_clustering).

The implementation is inspired by `scipy`'s UnionFind module, and it relies on `numba` for performance.

Installing
----------

Use `pip install fufpy`, or `pip install .` from the root directory of the project.

Dependencies
------------

This package depends on `numpy` and `numba`, which will be installed automatically when installing via `pip`.

License
-------

This software is published under the 3-clause BSD license.


Contents
--------

.. toctree::
    :maxdepth: 2

    usage
    api
