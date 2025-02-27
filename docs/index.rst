FufPy
=====


Implementation of union-find (aka `disjoint-set <https://en.wikipedia.org/wiki/Disjoint-set_data_structure>`_) data structure.
Currently, for performance, the structure is defined on a set :math:`\{0, \dots, n-1\}`, of size :math:`n`, which is specified at initialization.

It implements the standard operations, as well as :math:`subset`, which returns the subset corresponding to an element.
A main use case is `hierarchical clustering <https://en.wikipedia.org/wiki/Hierarchical_clustering>`_.

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

    api
