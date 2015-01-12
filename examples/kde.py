import geoplotlib
from geoplotlib.utils import read_csv, BoundingBox, DataAccessObject

#data = read_csv('somedata.csv')
#geoplotlib.kde(data, bw=[5,5], clip_above=100)
#geoplotlib.set_bbox(BoundingBox.KBH)

data = read_csv('somedata.csv')
geoplotlib.kde(data, bw=[5,5], clip_above=10)
geoplotlib.set_bbox(BoundingBox.DTU)

geoplotlib.show()
