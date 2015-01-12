import geoplotlib
from geoplotlib.utils import read_csv, BoundingBox


data = read_csv('somedata.csv')
geoplotlib.voronoi(data, line_color=[0,0,255])
geoplotlib.set_smoothing(True)
geoplotlib.set_bbox(BoundingBox.KBH)
geoplotlib.show()
