import geoplotlib
from geoplotlib.utils import read_csv, BoundingBox


data = read_csv('somedata.csv')
geoplotlib.voronoi(data, cmap='hot', max_area=1e4)
geoplotlib.set_bbox(BoundingBox.KBH)
geoplotlib.show()
