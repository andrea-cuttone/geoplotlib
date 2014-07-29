import sys,os
from pyglet.gl import glPushMatrix, glPopMatrix
from geoplotlib.layers import HotspotManager, PolyLayer
from geoplotlib.utils import parse_raw_str, BoundingBox

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
