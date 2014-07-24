import sys,os
sys.path.append(os.path.realpath('..'))

import geoplotlib
from geoplotlib.utils import read_csv


data = read_csv('./data/flights.csv')
geoplotlib.graph(lat0=data['lat. departure (decimal)'],
                 lon0=data['long. departure (decimal)'],
                 lat1=data['lat. arrival (decimal)'],
                 lon1=data['long. arrival (decimal)'],
                 linewidth=3.0,
                 color=[0,0,255,6]
                 )
geoplotlib.show()
