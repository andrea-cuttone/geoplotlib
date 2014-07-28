import sys,os
sys.path.append(os.path.realpath('..'))

import geoplotlib
from geoplotlib.utils import read_csv


data = read_csv('/Users/ancu/Dropbox/phd/code-projects/geoplotlib/examples/data/s-tog.csv')
geoplotlib.voronoi(data, f_tooltip=lambda r: r['name'])
geoplotlib.show()
