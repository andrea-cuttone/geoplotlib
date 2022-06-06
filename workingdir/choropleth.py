"""
선택된 나라의 실업률을 색깔로 바꾸는 작업=> 선택된 지역의 빈도수를 색깔로 바꾸는 작업으로
Based on by Mike Bostock's unemployment choropleth http://bl.ocks.org/mbostock/4060606
여기서는 geojson 파일들을 이용해야 한다..
"""

import geoplotlib
from geoplotlib.utils import BoundingBox
from geoplotlib.colors import ColorMap
import json


# find the unemployment rate for the selected county, and convert it to color
def get_color(properties):
    key = str(int(properties['STATE'])) + properties['COUNTY']
    if key in unemployment:
        return cmap.to_color(unemployment.get(key), .15, 'lin')
    else:
        return [0, 0, 0, 0]


with open('data/unemployment.json') as fin:
    unemployment = json.load(fin)

cmap = ColorMap('Blues', alpha=255, levels=10)
geoplotlib.geojson('data/gz_2010_us_050_00_20m.json', fill=True, color=get_color, f_tooltip=lambda properties: properties['NAME'])
geoplotlib.geojson('data/gz_2010_us_050_00_20m.json', fill=False, color=[255, 255, 255, 64])
geoplotlib.set_bbox(BoundingBox.WORLD)
geoplotlib.show()
