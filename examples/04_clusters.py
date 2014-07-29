import sys,os
sys.path.append(os.path.realpath('..'))

import geoplotlib
from geoplotlib.utils import read_csv, BoundingBox


data = read_csv('./data/loc-andrea-prod-resampled.csv')
data = {'lon': data['lon'][:10000], 'lat': data['lat'][:10000]}
geoplotlib.clusters(data)
geoplotlib.set_bbox(BoundingBox.KBH)
geoplotlib.show()
