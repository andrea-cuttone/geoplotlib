import geoplotlib
from geoplotlib.utils import read_csv, BoundingBox


data = read_csv('./data/stops-andrea.csv')
labels = set(data['label'])
cols = geoplotlib.colors.create_set_cmap(labels, 'hsv')
for l in labels:
    geoplotlib.scatter(data.where(data['label'] == l), color=cols[l])

#geoplotlib.scatter(data, color=[255,0,0])

geoplotlib.set_bbox(BoundingBox(north=55.690196,west=12.544253,south=55.671,east=12.599))
geoplotlib.show()
