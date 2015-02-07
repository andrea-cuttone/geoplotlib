import geoplotlib
from geoplotlib.utils import read_csv, BoundingBox


data = read_csv('data/bus.csv')
geoplotlib.voronoi(data, cmap='Blues_r', max_area=1e3)
geoplotlib.set_smoothing(True)
geoplotlib.set_bbox(BoundingBox.DK)
geoplotlib.show()
