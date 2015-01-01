import geoplotlib
from geoplotlib.utils import read_csv


data = read_csv('somedata.csv')
geoplotlib.scatter(data)
geoplotlib.show()
