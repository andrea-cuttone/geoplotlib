from geoplotlib.utils import BoundingBox
import geoplotlib


geoplotlib.shapefiles('/Users/ancu/denmark-latest.shp/buildings',
                  shape_type='full',
                   f_tooltip=lambda attr: attr['name'])

geoplotlib.set_bbox(BoundingBox.KBH)
geoplotlib.set_smoothing(True)
geoplotlib.show()
