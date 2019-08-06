from copy import deepcopy
from os import environ
from subprocess import check_output
from sys import executable

setup_image = """
import numpy as np
from mgen import rotation_from_angles
from scipy.ndimage import affine_transform

from affine_transform import transform

input_image = np.random.rand(*{image_shape})
translate = np.asarray({translate})
rotate = np.asarray({rotation})
center_of_rotation = np.array({center_of_rotation})
output = np.zeros({image_shape})
order = "{order}"

"""

_setup_this = """"""

_setup_scipy = """rotate = rotate.T
offset = center_of_rotation - rotate.dot(center_of_rotation + translate)

if order == "cubic":
    order = 3
elif order == "linear":
    order = 1
else:
    raise RuntimeError("Invalid order selected.")
"""


def _setup(**scenario):
    return setup_image.format(**scenario)


def setup_scipy(**scenario):
    return _setup(**scenario) + _setup_scipy


def setup_this(**scenario):
    return _setup(**scenario) + _setup_this


def _benchmark(code, setup, num_threads):
    setup_script = setup.replace("\n", "\\n").replace('"', '\\"')

    env = deepcopy(environ)
    if num_threads > 1:
        env["OMP_NUM_THREADS"] = f"{num_threads}"
    return float(
        check_output(
            [
                executable,
                "-c",
                f'import timeit\nprint(min(timeit.Timer("{code}", setup="{setup_script}").repeat(2,1)))',
            ],
            env=env,
        )
    )


def benchmark_scipy_affine_transform(num_threads, **scenario):
    return _benchmark(
        "affine_transform(input_image, rotate, offset=offset, output=output, order=order)",
        setup_scipy(**scenario),
        num_threads,
    )


def benchmark_affine_transform(num_threads, **scenario):
    return _benchmark(
        "transform(input_image, rotate, translate, origin=center_of_rotation, output_image=output, order=order)",
        setup_this(**scenario),
        num_threads,
    )


def run_benchmark(num_threads, scenarios):
    return_vals = {}

    return_vals["self"] = [
        benchmark_affine_transform(num_threads, **scenario) for scenario in scenarios
    ]

    if num_threads == 1:
        return_vals["scipy"] = [
            benchmark_scipy_affine_transform(num_threads, **scenario)
            for scenario in scenarios
        ]

    return return_vals
