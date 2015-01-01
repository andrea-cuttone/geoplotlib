import geoplotlib
from geoplotlib.utils import read_csv, BoundingBox


data = read_csv('somedata.csv')
geoplotlib.kde(data, bw=[4,4], method='hist', scaling='log', scalemin=0.1, scalemax=.8)
#geoplotlib.scatter(data, color=[0,0,255])
geoplotlib.set_smoothing(False)
geoplotlib.set_bbox(BoundingBox.KBH)
geoplotlib.show()
