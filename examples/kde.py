"""
Multiple examples of kernel density estimation visualization
"""
import geoplotlib
from geoplotlib.utils import read_csv, BoundingBox, DataAccessObject

data = read_csv('data/opencellid_dk.csv')

geoplotlib.kde(data, bw=[5,5], cut_below=1e-6)

# lowering clip_above changes the max value in the color scale
#geoplotlib.kde(data, bw=[5,5], cut_below=1e-6, clip_above=1)

# different bandwidths
#geoplotlib.kde(data, bw=[20,20], cmap='coolwarm', cut_below=1e-6)
#geoplotlib.kde(data, bw=[2,2], cmap='coolwarm', cut_below=1e-6)

# linear colorscale
#geoplotlib.kde(data, bw=[5,5], cmap='jet', cut_below=1e-6, scaling='lin')

geoplotlib.set_bbox(BoundingBox.DK)
geoplotlib.show()
