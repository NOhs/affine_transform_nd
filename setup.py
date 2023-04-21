import os
import re
import sys
import platform
import subprocess

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        if platform.system() == "Windows":
            cmake_version = LooseVersion(
                re.search(r"version\s*([\d.]+)", out.decode()).group(1)
            )
            if cmake_version < "3.9.0":
                raise RuntimeError("CMake >= 3.9.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = (
            os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
            + "/affine_transform"
        ).replace("\\", "/")
        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DPYTHON_EXECUTABLE=" + sys.executable,
        ]

        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        if platform.system() == "Windows":
            cmake_args += [
                "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir)
            ]
            if sys.maxsize > 2 ** 32:
                cmake_args += ["-A", "x64"]
            build_args += ["--", "/m"]
        else:
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
            build_args += ["--", "-j2"]

        env = os.environ.copy()
        # env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
        #                                                     self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )


this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.rst"), encoding="utf-8") as f:
    long_description = f.read()

from setuptools_scm import get_version
from datetime import datetime

try:
    # If in git repository, get git label
    v = get_version(root='.', relative_to=__file__)

    if "+" in v:
        v = v.split("+")[0] + datetime.today().strftime("%Y%m%d%H%M%S")

    with open(
        os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "affine_transform",
            "version.txt",
        ),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(v)
except LookupError:
    # Otherwise get version from version.txt (sdist for example)
    with open(
        os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "affine_transform",
            "version.txt",
        ),
        encoding="utf-8",
    ) as f:
        v = f.read()

setup(
    name="affine_transform",
    author="NOhs, TobelRunner",
    version=v,
    url="https://github.com/NOhs/affine_transform_nd",
    description="Easy to use multi-core affine transformations",
    python_requires='>=3.8',
    long_description=long_description,
    license="MIT",
    ext_modules=[CMakeExtension("affine_transform")],
    package_data={"affine_transform": ["version.txt"]},
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: C++",
        "License :: OSI Approved :: MIT License",
    ],
    packages=find_packages(),
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    install_requires=["numpy", "mgen", "setuptools_scm"],
    setup_requires=["pytest-runner", "setuptools_scm"],
    tests_require=["numpy", "mgen", "pytest"],
)
