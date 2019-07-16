Affine transformation
=====================

.. toctree::

The affine transformation is implemented by taking an origin and
a basis for the new transformed image. This coordinate system is
then put on the input image and nested loops iterate along all given
basis vectors. The outer loop, that is the iteration along the first
basis vector, is OpenMP parallelized. The n-dimensional
loop is realized via a variadic template loop function. It uses all
the functionality defined in the interpolation section to help extracting
data from the given input image.

.. toctree::
    :maxdepth: 2
    :caption: C++ definitions

    transform
    transform_loop