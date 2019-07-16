Interpolation
=============

.. toctree::

One way to implement n-dimensional interpolation is to use the
fact that an n-dimensional interpolation can be expressed as a
1-dimensional interpolation of (n-1)-dimensional interpolations.

Therefore, we first can defined one dimensional interpolation functions
as all higher dimensional interpolations can be expressed as multiple
applications of these one dimensional functions.

.. toctree::
    :maxdepth: 2

    linear_interpolation
    cubic_interpolation

These interpolation functions have static attributes that tell you
how many data-points they need to operate. As an input they take
the ``Data`` struct with the dimension set to 1. This ``Data`` struct
is an n-dimensional array (internally using ``std::array``).

.. toctree::
    :maxdepth: 2

    data_struct

One also needs to decide how to deal with the image boundaries. There are
various ways to do this. Currently, only the following way is implemented.

.. toctree::
    :maxdepth: 2

    constant_boundary

Finally, two functions are defined that realise the nD interpolation.
One is to extract the necessary data from a given image, the other one is
to take a given position and interpolate that data.

.. toctree::
    :maxdepth: 2

    extract
    apply_interpolation