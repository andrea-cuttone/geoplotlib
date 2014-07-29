from geoplotlib.core import BaseApp
from geoplotlib.layers import ScatterLayer, HistogramLayer, GraphLayer, ClusterLayer, PolyLayer, VoronoiLayer
import numpy as np


_app = BaseApp()


def show():
    _app.run()


def scatter(data, **kwargs):
    _app.add_layer(ScatterLayer(data, **kwargs))


def hist(data):
    _app.add_layer(HistogramLayer(data))


def graph(src_lat, src_lon, dest_lat, dest_lon, **kwargs):
    _app.add_layer(GraphLayer(src_lat, src_lon, dest_lat, dest_lon, **kwargs))


def clusters(data):
    _app.add_layer(ClusterLayer(data))


def clear():
    _app.clear_layers()


def tiles_provider(tiles_provider):
    _app.map_layer.tiles_provider = tiles_provider


def add_layer(layer):
    _app.add_layer(layer)


def shapefiles(fname, **kwargs):
    _app.add_layer(PolyLayer(fname, **kwargs))


def voronoi(data, **kwargs):
    _app.add_layer(VoronoiLayer(data, **kwargs))


def set_bbox(bbox):
    _app.set_bbox(bbox)
