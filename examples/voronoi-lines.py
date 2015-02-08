"""
Example of voronoi layer with lines
"""
import geoplotlib
from geoplotlib.utils import read_csv, BoundingBox


data = read_csv('data/bus.csv')
geoplotlib.voronoi(data, line_color='b')
geoplotlib.set_smoothing(True)
geoplotlib.set_bbox(BoundingBox.DK)
geoplotlib.show()
