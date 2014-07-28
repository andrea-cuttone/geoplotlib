import sys,os
sys.path.append(os.path.realpath('..'))

import geoplotlib
from geoplotlib.utils import read_csv


metro = read_csv('./data/metro.csv')
s_tog = read_csv('./data/s-tog.csv')
geoplotlib.scatter(metro,
                   color=[0,0,255],
                   f_tooltip=lambda r: r['name'])
geoplotlib.scatter(s_tog,
                   color=[255,0,0],
                   f_tooltip=lambda r: r['name'])
geoplotlib.show()
