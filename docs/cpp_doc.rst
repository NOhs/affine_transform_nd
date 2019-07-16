C++ Code Documentation
======================

This C++ documentation is not meant as a public API documentation,
but rather as a behind the scene implementation documentation for
those that are interested in how it is implemented, or how it could
be extended/modified.

General structure
-----------------

Affine transformations can be split into two tasks:

- Iterating over a grid to fill the output image
- Interpolating while looking up values at each grid point

The C++ code is thus structured in this fashion. Furthermore,
one implementation file binds the functions defined in these headers
to python.

.. toctree::
    :maxdepth: 2
    :caption: C++ sections

    cpp_docs/interpolation
    cpp_docs/affine_transform
    cpp_docs/python_bindings


