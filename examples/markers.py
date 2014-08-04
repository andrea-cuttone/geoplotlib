import sys,os
import pyglet
from geoplotlib.core import BatchPainter
from geoplotlib.layers import BaseLayer, HotspotManager, ScatterLayer

sys.path.append(os.path.realpath('..'))

import geoplotlib
from geoplotlib.utils import read_csv


metro = read_csv('./data/metro.csv')
s_tog = read_csv('./data/s-tog.csv')

geoplotlib.markers(metro, 'data/m.png', f_tooltip=lambda r: r['name'])
geoplotlib.markers(s_tog, 'data/s-tog.png', f_tooltip=lambda r: r['name'])
geoplotlib.show()
