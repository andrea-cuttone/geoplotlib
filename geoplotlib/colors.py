from random import shuffle
import math


def _convert_color_format(col, alpha):
    return [int(c * 255) for c in col[:3]] + [alpha]


class ColorMap():

    def __init__(self, cmap_name, alpha=255, levels=10):
        """
        Converts continuous values into colors using matplotlib colorscales
        :param cmap_name: colormap name
        :param alpha: color alpha
        :param levels: discretize the colorscale into levels
        """
        from pylab import get_cmap
        self.cmap = get_cmap(cmap_name)
        self.alpha = alpha
        self.levels = levels
        self.mapping = {}


    def to_color(self, value, maxvalue, scale):
        """
        convert continuous values into colors using matplotlib colorscales
        :param value: value to be converted
        :param maxvalue: max value in the colorscale
        :param scale: lin, log, sqrt
        :return: the color corresponding to the value
        """
        if value < 0 or maxvalue < 0:
            raise Exception('no negative values allowed')

        if scale == 'lin':
            if maxvalue == 0:
                value = 0
            else:
                value = value / maxvalue
        elif scale == 'log':
            if value < 1 or maxvalue <= 1:
                raise Exception('values must be >= 1')
            else:
                value = math.log(value) / math.log(maxvalue)
        elif scale == 'sqrt':
            if maxvalue == 0:
                value = 0
            else:
                value = math.sqrt(value) / math.sqrt(maxvalue)
        elif scale == 'fifthroot':
            if maxvalue == 0:
                value = 0
            else:
                value = value**0.2 / maxvalue**0.2
        else:
            raise Exception('scale must be lin, log, sqrt or fifthroot')

        value = min(value,1)
        delta = 1. / self.levels
        value = round(value / delta) * delta
        if value not in self.mapping:
            self.mapping[value] = _convert_color_format(self.cmap(value), self.alpha)
        return self.mapping[value]


def create_set_cmap(values, cmap_name, alpha=255):
    """
    return a dict of colors corresponding to the unique values
    :param values: values to be mapped
    :param cmap_name: colormap name
    :param alpha: color alpha
    :return: dict of colors corresponding to the unique values
    """
    unique_values = list(set(values))
    shuffle(unique_values)
    from pylab import get_cmap
    cmap = get_cmap(cmap_name)
    d = {}
    for i in range(len(unique_values)):
        d[unique_values[i]] = _convert_color_format(cmap(1.*i/len(unique_values)), alpha)
    return d


def colorbrewer(values, alpha=255):
    """
    Return a dict of colors for the unique values.
    Colors are adapted from Harrower, Mark, and Cynthia A. Brewer.
    "ColorBrewer. org: an online tool for selecting colour schemes for maps."
    The Cartographic Journal 40.1 (2003): 27-37.

    :param values: values
    :param alpha: color alphs
    :return: dict of colors for the unique values.
    """
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
