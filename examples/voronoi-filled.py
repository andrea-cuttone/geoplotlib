"""
Example of voronoi layer with filled area
"""
import geoplotlib
from geoplotlib.utils import read_csv, BoundingBox


data = read_csv('data/bus.csv')
geoplotlib.voronoi(data, cmap='Blues_r', max_area=1e5, alpha=255, f_tooltip=lambda d:d['name'])
geoplotlib.set_bbox(BoundingBox.DK)
geoplotlib.show()
