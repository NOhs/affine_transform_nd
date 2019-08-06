import inspect
import sys
from argparse import ArgumentParser
from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt
from mgen import rotation_from_angles
from pandas import DataFrame
from tqdm import tqdm

from _benchmark_misc import run_benchmark

plt.style.use("fivethirtyeight")

COLOR = '#404040'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR
plt.rcParams['xtick.color'] = COLOR
plt.rcParams['ytick.color'] = COLOR

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Lato",
    "proxima-nova",
    "Helvetica Neue",
    "Arial",
    "sans-serif",
]

default_scenario = {
    "image_shape": [500] * 3,
    "translate": [50, 100, -10],
    "rotation": 'rotation_from_angles((np.pi / 7, np.pi / 3 * 2, np.pi / 11), "XYZ")',
    "center_of_rotation": [249.5] * 3,
    "order": "cubic",
}


def _scenario_image_size(shape_func, plot_label, x_axis_label, filename):
    image_size = [10, 25, 50, 75, 100, 250, 500, 750, 1000]
    center_of_rotation = [((size - 1) / 2,) * 3 for size in image_size]
    image_shapes = [shape_func(i) for i in image_size]

    x_labels = [np.prod(size) for size in image_shapes]

    scenarios = []
    for shape, center in zip(image_shapes, center_of_rotation):
        new_scenario = deepcopy(default_scenario)
        new_scenario["image_shape"] = shape
        new_scenario["center_of_rotation"] = center

        scenarios.append(new_scenario)

    for order in ("linear", "cubic"):
        total_vals = {}
        for scenario in scenarios:
            scenario["order"] = order

        num_threads = [1,4,8]

        for multicore in num_threads:
            total_vals[multicore] = run_benchmark(multicore, scenarios)

        fig, ax = plt.subplots()
        ax.loglog(x_labels, total_vals[1]["scipy"], label="scipy - 1 thread")
        for i in num_threads:
            ax.loglog(x_labels, total_vals[i]["self"], label=f"self - {i} thread{'s' if i > 1 else ''}")

        ax.grid(True, which="major")

        #ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

        # _plot_label = plot_label + f" / {order} interpolation"

        _filename = filename + "_" + order

        # ax.set_title(_plot_label)
        ax.set_xlabel(x_axis_label)
        ax.set_ylabel("time [s]")
        ax.tick_params(axis="both", which="major")
        background_color = "#FCFCFC"
        ax.set_facecolor(background_color)
        ax.spines['bottom'].set_color(background_color)
        ax.spines['top'].set_color(background_color)
        ax.spines['right'].set_color(background_color)
        ax.spines['left'].set_color(background_color)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
          framealpha=1, facecolor=background_color)
        #plt.tight_layout()
        plt.savefig(_filename + ".svg", bbox_inches="tight", facecolor=background_color)


def scenario_different_image_sizes():
    _scenario_image_size(
        lambda i: (i,) * 3, "image dimension scaling", "# voxels", "image_scaling"
    )


# def scenario_different_image_sizes_x():
#    _scenario_image_size(
#        lambda i: (i, 500, 500),
#        "image x-dimension scaling",
#        "# voxels",
#        "image_scaling_x",
#    )
#
#
# def scenario_different_image_sizes_y():
#    _scenario_image_size(
#        lambda i: (500, i, 500),
#        "image y-dimension scaling",
#        "# voxels",
#        "image_scaling_y",
#    )
#
#
# def scenario_different_image_sizes_z():
#    _scenario_image_size(
#        lambda i: (500, 500, i),
#        "image z-dimension scaling",
#        "# voxels",
#        "image_scaling_z",
#    )
#

if __name__ == "__main__":
    benchmarkfunctions = [
        obj
        for name, obj in inspect.getmembers(sys.modules[__name__])
        if (inspect.isfunction(obj) and name.startswith("scenario"))
    ]

    for func in tqdm(benchmarkfunctions):
        func()
