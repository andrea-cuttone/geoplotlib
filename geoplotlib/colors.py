from random import shuffle
from math import log
import pylab as plt


def _convert_color_format(col, alpha):
    return [int(c * 255) for c in col[:3]] + [alpha]


def create_set_cmap(values, cmap_name, alpha=255):
    unique_values = list(set(values))
    #shuffle(unique_values)
    cmap = plt.get_cmap(cmap_name)
    d = {}
    for i in range(len(unique_values)):
        d[unique_values[i]] = _convert_color_format(cmap(1.*i/len(unique_values)), alpha)
    return d


def create_linear_cmap(vmax, cmap_name, vmin=0, alpha=255):
    cmap = plt.get_cmap(cmap_name)
    return lambda x: _convert_color_format(cmap(1.*(x - vmin)/(vmax - vmin)), alpha)


def create_log_cmap(vmax, cmap_name, vmin=1, alpha=255):
    cmap = plt.get_cmap(cmap_name)
    return lambda x: _convert_color_format(cmap(log(x)/log(vmax)), alpha)


def colorbrewer(values, alpha=255):
    basecolors = [
        [31, 120, 180],
        [178, 223, 138],
        [51, 160, 44],
        [251, 154, 153],
        [227, 26, 28],
        [253, 191, 111],
        [255, 127, 0],
        [202, 178, 214],
        [106, 61, 154],
        [255, 255, 153],
        [177, 89, 40]
    ]
    unique_values = list(set(values))
    return {k: basecolors[i % len(basecolors)] + [alpha]  for i, k in enumerate(unique_values)}
