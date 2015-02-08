"""
Example of 2D histogram
"""
import geoplotlib
from geoplotlib.utils import read_csv, BoundingBox


data = read_csv('data/opencellid_dk.csv')
geoplotlib.hist(data, colorscale='sqrt', binsize=8)
geoplotlib.set_bbox(BoundingBox.DK)
geoplotlib.show()
