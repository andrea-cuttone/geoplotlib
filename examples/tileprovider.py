import geoplotlib
from geoplotlib.utils import read_csv


data = read_csv('./data/loc-andrea-prod-resampled.csv')
geoplotlib.tiles_provider(lambda zoom, xtile, ytile: 'http://a.tile.stamen.com/watercolor/%d/%d/%d.png' % (zoom, xtile, ytile))
geoplotlib.scatter(data)
geoplotlib.show()
