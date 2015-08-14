import geoplotlib
from geoplotlib.colors import colorbrewer
from geoplotlib.utils import epoch_to_str, BoundingBox, read_csv


data = read_csv('./data/metro.csv')
geoplotlib.dot(data, 'r')
geoplotlib.labels(data, 'name', color=[0,0,255,255], font_size=10, anchor_x='center')
geoplotlib.set_bbox(BoundingBox.KBH)
geoplotlib.show()
