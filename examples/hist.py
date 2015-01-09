import geoplotlib
from geoplotlib.utils import read_csv, BoundingBox


data = read_csv('somedata.csv')
geoplotlib.hist(data, logscale=True, binsize=8, binscaling=False)
geoplotlib.show()
