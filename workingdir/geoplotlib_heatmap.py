import geoplotlib
from geoplotlib.utils import read_csv, BoundingBox, DataAccessObject
import pandas as pd

df = pd.read_csv('무장애여행정보.csv')
df.columns = ['State', 'District', '위도', '경도']
geoplotlib.kde(df, bw=8, cut_below=1e+4)

geoplotlib.set_bbox(BoundingBox.jeju)
geoplotlib.show()
