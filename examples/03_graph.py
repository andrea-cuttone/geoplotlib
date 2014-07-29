import sys,os
sys.path.append(os.path.realpath('..'))

import geoplotlib
from geoplotlib.utils import read_csv


data = read_csv('./data/flights.csv')
geoplotlib.graph(src_lat=data['lat. departure (decimal)'],
                 src_lon=data['long. departure (decimal)'],
                 dest_lat=data['lat. arrival (decimal)'],
                 dest_lon=data['long. arrival (decimal)'])
geoplotlib.show()
