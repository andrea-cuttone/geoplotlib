"""
Example of voronoi layer with lines
"""
import geoplotlib
from geoplotlib.utils import read_csv, BoundingBox


data = read_csv('data/bus.csv')
geoplotlib.voronoi(data, line_color='b', f_tooltip=lambda d:d['name'], line_width=1)
geoplotlib.set_smoothing(True)
geoplotlib.set_bbox(BoundingBox.DK)
geoplotlib.show()
