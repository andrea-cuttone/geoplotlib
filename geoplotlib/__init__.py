from geoplotlib.core import BaseApp
from geoplotlib.layers import ScatterLayer, HistogramLayer, GraphLayer, ClusterLayer, PolyLayer, VoronoiLayer, \
    MarkersLayer
import numpy as np


_app = BaseApp()


def show():
    _app.run()


def scatter(data, color=None, point_size=6, f_tooltip=None):
    _app.add_layer(ScatterLayer(data, color=color, point_size=point_size, f_tooltip=f_tooltip))


def hist(data, cmap='coolwarm', alpha=220, binsize=32, show_tooltip=True, vmin=1, f_group=None):
    _app.add_layer(HistogramLayer(data, cmap=cmap, alpha=alpha, binsize=binsize, show_tooltip=show_tooltip, vmin=vmin, f_group=f_group))


def graph(data, src_lat, src_lon, dest_lat, dest_lon, **kwargs):
    _app.add_layer(GraphLayer(data, src_lat, src_lon, dest_lat, dest_lon, **kwargs))


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


def markers(data, marker, **kwargs):
    _app.add_layer(MarkersLayer(data, marker, **kwargs))


def set_bbox(bbox):
    _app.set_bbox(bbox)
