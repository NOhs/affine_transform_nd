Affine Transformation: C++17, OpenMP, Python
============================================

|test| |appveyor| |codecov| |rtd| |pypi| |python_vers| |GCC| |license| |codacy| |black|


.. |test| image:: https://github.com/NOhs/mgen/actions/workflows/test.yml/badge.svg
    :target: https://github.com/NOhs/mgen/actions/workflows/test.yml
    :alt: Test Status
.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/bh3gsedf83576wus/branch/master?svg=true
    :target: https://ci.appveyor.com/project/NOhs/affine-transform-nd/branch/master
    :alt: AppVeyor Status
.. |codecov| image:: https://codecov.io/gh/NOhs/affine_transform_nd/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/NOhs/affine_transform_nd
    :alt: Codecov Status
.. |rtd| image:: https://readthedocs.org/projects/affine-transform-nd/badge/?version=latest
    :target: https://affine-transform-nd.readthedocs.io/en/latest/?badge=latest
    :alt: ReadTheDocs Status
.. |pypi| image:: https://img.shields.io/pypi/v/affine_transform.svg?color=dark%20green
    :target: https://pypi.org/project/affine_transform
    :alt: PyPI
.. |python_vers| image:: https://img.shields.io/pypi/pyversions/affine_transform
    :alt: PyPI - Python Version
.. |GCC| image:: https://img.shields.io/badge/GCC-9%20|%2010%20|%2011%20|%2012-blue
    :alt: Compiler Version
.. |license| image:: https://img.shields.io/github/license/NOhs/affine_transform_nd.svg?color=blue
    :target: https://opensource.org/licenses/MIT
    :alt: license
.. |codacy| image:: https://api.codacy.com/project/badge/Grade/e39c4c5b913d4237b77fa07f679ab521
    :target: https://www.codacy.com/app/NOhs/affine_transform_nd?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=NOhs/affine_transform_nd&amp;utm_campaign=Badge_Grade
    :alt: code quality
.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/ambv/black
    :alt: code style

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
- Arbitrary shaped output arrays, allowing e.g. to only extract a transformed slice

Short example usage
-------------------

.. code-block:: python

    import numpy as np

    from affine_transform import transform
    from mgen import rotation_from_angle


    # Create a simple white square in an image
    original = np.zeros((601, 401))
    original[100:300, 100:300] = 1

    # Rotate by 22.5° (around the centre of the square (200,200))
    # and shift +200 in x and +100 in y
    transformed = transform(
        original, rotation_from_angle(np.pi / 8), np.array([200, 100]), origin=(200, 200)
    )
