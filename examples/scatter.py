import geoplotlib
from geoplotlib.utils import read_csv


data = read_csv('data/loc-andrea-prod-resampled.csv')
geoplotlib.scatter(data)
geoplotlib.show()
