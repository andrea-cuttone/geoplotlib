import sys,os
from geoplotlib.utils import BoundingBox

sys.path.append(os.path.realpath('..'))

import geoplotlib


geoplotlib.shapefiles('data/dk_kommune/dk_kommune',
                  f_tooltip=lambda attr: attr['STEDNAVN'],
                  color=[0,0,255])

# geoplotlib.shapes('/Users/ancu/denmark-latest.shp/buildings',
#                   shape_type='full',
#                   f_tooltip=lambda attr: attr['name'])

geoplotlib.set_bbox(BoundingBox.DK)
geoplotlib.show()
