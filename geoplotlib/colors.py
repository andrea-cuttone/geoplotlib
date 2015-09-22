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


    def to_color(self, value, maxvalue, scale, minvalue=0.0):
        """
        convert continuous values into colors using matplotlib colorscales
        :param value: value to be converted
        :param maxvalue: max value in the colorscale
        :param scale: lin, log, sqrt
        :param minvalue: minimum of the input values in linear scale (default is 0)
        :return: the color corresponding to the value
        """
        
        if scale == 'lin':
            if minvalue >= maxvalue:
                raise Exception('minvalue must be less than maxvalue')
            else:
                value = 1.*(value-minvalue) / (maxvalue-minvalue)
        elif scale == 'log':
            if value < 1 or maxvalue <= 1:
                raise Exception('value and maxvalue must be >= 1')
            else:
                value = math.log(value) / math.log(maxvalue)
        elif scale == 'sqrt':
            if value < 0 or maxvalue <= 0:
                raise Exception('value and maxvalue must be greater than 0')
            else:
                value = math.sqrt(value) / math.sqrt(maxvalue)
        else:
            raise Exception('scale must be "lin", "log", or "sqrt"')

        if value < 0:
            value = 0
        elif value > 1:
            value = 1

        value = int(1.*self.levels*value)*1./(self.levels-1)

        if value not in self.mapping:
            self.mapping[value] = _convert_color_format(self.cmap(value), self.alpha)
        return self.mapping[value]


    def get_boundaries(self, maxvalue, scale):
        edges = []
        colors = []
        
        for i in range(self.levels+1):
            z = 1. * i / self.levels
            if scale == 'lin':
                edges.append(maxvalue * z)
            elif scale == 'log':
                # log(v)/log(maxvalue) = z
                # log(v) = log(maxvalue)*z
                # v = e^(log(maxvalue)*z)
                edges.append(math.exp(math.log(maxvalue)*z))
            elif scale == 'sqrt':
                # sqrt(v)/sqrt(maxvalue) = z
                # v/maxvalue = z^2
                # v = maxvalue * z^2
                edges.append(maxvalue * z**2)
            else:
                raise Exception('scale must be "lin", "log", or "sqrt"') 

        for e in edges[:-1]:
            colors.append(self.to_color(e, maxvalue, scale))

        return edges, colors


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
