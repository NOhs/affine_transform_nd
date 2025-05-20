from os import path as _path

from .affine_transform import transform

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.0"