from os import path as _path

from .affine_transform import transform

with open(
    _path.join(_path.abspath(_path.dirname(__file__)), "version.txt"), encoding="utf-8"
) as f:
    __version__ = f.read()
