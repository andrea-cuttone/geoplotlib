import geoplotlib
from geoplotlib.utils import read_csv, BoundingBox


data = read_csv('somedata.csv').head(10000)
geoplotlib.kde(data, bw=[10,10], binsize=10, vmin=0.01, vmax=1)
geoplotlib.set_bbox(BoundingBox.DTU)
geoplotlib.show()
