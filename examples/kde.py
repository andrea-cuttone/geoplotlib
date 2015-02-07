import geoplotlib
from geoplotlib.utils import read_csv, BoundingBox, DataAccessObject

data = read_csv('data/opencellid_dk.csv')
geoplotlib.kde(data, bw=[5,5], cut_below=1e-6)
geoplotlib.set_bbox(BoundingBox.DK)

geoplotlib.show()
