import sys,os
sys.path.append(os.path.realpath('..'))

import geoplotlib
from geoplotlib.utils import read_csv


data = read_csv('./data/loc-andrea-prod-resampled.csv')
data = {'lon': data['lon'][:10000], 'lat': data['lat'][:10000]}
geoplotlib.clusters(data)
geoplotlib.show()
