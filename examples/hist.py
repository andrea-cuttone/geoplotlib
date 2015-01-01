import geoplotlib
from geoplotlib.utils import read_csv, BoundingBox


data = read_csv('somedata.csv')
geoplotlib.hist(data, logscale=True, binsize=32, binscaling=False)
geoplotlib.show()
