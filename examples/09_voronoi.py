import geoplotlib
from geoplotlib.utils import read_csv


data = read_csv('/Users/ancu/Dropbox/phd/code-projects/geoplotlib/examples/data/bus.csv')
geoplotlib.voronoi(data, f_tooltip=lambda r: r['name'])
geoplotlib.show()
