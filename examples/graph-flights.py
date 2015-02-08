"""
Example of spatial graph
"""
import geoplotlib
from geoplotlib.utils import read_csv


data = read_csv('./data/flights.csv')
geoplotlib.graph(data,
                 src_lat='lat. departure (decimal)',
                 src_lon='long. departure (decimal)',
                 dest_lat='lat. arrival (decimal)',
                 dest_lon='long. arrival (decimal)',
                 color='hot_r',
                 alpha=16,
                 linewidth=2)
geoplotlib.show()
