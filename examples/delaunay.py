"""
Example of delaunay triangulation
"""
from geoplotlib.layers import DelaunayLayer
import geoplotlib
from geoplotlib.utils import read_csv, BoundingBox


data = read_csv('data/bus.csv')
geoplotlib.delaunay(data, cmap='hot_r')
geoplotlib.set_bbox(BoundingBox.DK)
geoplotlib.set_smoothing(True)
geoplotlib.show()
