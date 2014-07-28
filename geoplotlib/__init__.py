from geoplotlib.core import BaseApp
from geoplotlib.layers import ScatterLayer, HistogramLayer, GraphLayer, ClusterLayer, PolyLayer


_app = BaseApp()


def show():
    _app.run()


def scatter(data, **kwargs):
    _app.add_layer(ScatterLayer(data, **kwargs))


def hist(data):
    _app.add_layer(HistogramLayer(data))


def graph(lat0, lon0, lat1, lon1, **kwargs):
    _app.add_layer(GraphLayer(lat0, lon0, lat1, lon1, **kwargs))


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
