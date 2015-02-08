"""
Example of scatterplot
"""
import geoplotlib
from geoplotlib.utils import read_csv


data = read_csv('data/bus.csv')
geoplotlib.scatter(data)
geoplotlib.show()
