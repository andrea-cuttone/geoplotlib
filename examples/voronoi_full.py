import geoplotlib
from geoplotlib.utils import read_csv, BoundingBox


data = read_csv('somedata.csv')
geoplotlib.voronoi(data, cmap='hot_r', max_area=1e4)
geoplotlib.set_bbox(BoundingBox.DTU)
geoplotlib.set_map_alpha(64)
geoplotlib.show()
