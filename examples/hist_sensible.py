import sys,os
sys.path.append(os.path.realpath('..'))

import geoplotlib
from geoplotlib.utils import BoundingBox, DataAccessObject


import pandas as pd
df = pd.read_csv('somedata.csv')

data = DataAccessObject({'lon': df.lon.values, 'lat': df.lat.values, 'user': df.user.values})
geoplotlib.hist(data,
                alpha=220,
                vmin=3,
                binsize=6,
                show_tooltip=False,
                f_group=lambda group_mask: len(set(data['user'][group_mask]))
                )
geoplotlib.set_bbox(BoundingBox.KBH)
geoplotlib.show()
