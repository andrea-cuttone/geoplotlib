import sys,os
from pyglet.gl import glPushMatrix, glPopMatrix
from geoplotlib.layers import HotspotManager, PolyLayer
from geoplotlib.utils import parse_raw_str

sys.path.append(os.path.realpath('..'))

import geoplotlib


geoplotlib.shapes('data/dk_kommune/dk_kommune', f_tooltip=lambda attr: attr['STEDNAVN'])
geoplotlib.show()
