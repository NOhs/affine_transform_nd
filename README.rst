Affine Transformation: C++17, OpenMP, Python
============================================

This project explores how C++17 and OpenMP can be combined to write a
surprisingly compact implementation of n-dimensional parallel affine
transformations which are linked into Python via the ``affine_transform``
module.

While this project is still under development, the following features
are supported:

- Linear and cubic (without prefiltering) interpolation
- Constant boundaries
- Compiling code for arbitrarily dimensional data
- Parallelism via OpenMP

Short example usage
-------------------

.. code-block:: python

    import numpy as np

    from affine_transform import transform
    from mgen import rotation_from_angle

    import matplotlib.pyplot as plt


    # Create a simple white square in an image
    original = np.zeros((601, 401))
    original[100:300, 100:300] = 1

    # Rotate by 22.5° (around the centre of the square (200,200))
    # and shift +200 in x and +100 in y
    transformed = transform(
        original, rotation_from_angle(np.pi / 8), np.array([200, 100]), origin=(200, 200)
    )
