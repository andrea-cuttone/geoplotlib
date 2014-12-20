import geoplotlib
from geoplotlib.utils import read_csv, BoundingBox


data = read_csv('/Users/ancu/Dropbox/phd/code-projects/geoplotlib/examples/data/bus.csv')
geoplotlib.voronoi(data, line_color=[0,0,255], f_tooltip=lambda r: r['name'])
geoplotlib.scatter(data, point_size=1)
geoplotlib.set_bbox(BoundingBox.KBH)
geoplotlib.show()
