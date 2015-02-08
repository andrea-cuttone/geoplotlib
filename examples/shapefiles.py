"""
Example of rendering shapefiles
"""
from geoplotlib.utils import BoundingBox
import geoplotlib


geoplotlib.shapefiles('data/dk_kommune/dk_kommune',
                  f_tooltip=lambda attr: attr['STEDNAVN'],
                  color=[0,0,255])

geoplotlib.set_smoothing(True)
geoplotlib.set_bbox(BoundingBox.DK)
geoplotlib.show()
