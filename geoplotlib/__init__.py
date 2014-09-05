from multiprocessing import Process
from geoplotlib.core import BaseApp
from geoplotlib.layers import ScatterLayer, HistogramLayer, GraphLayer, ClusterLayer, PolyLayer, VoronoiLayer, \
    MarkersLayer


class AppConfig:

    def __init__(self):
        self.reset()


    def reset(self):
        self.layers = []
        self.bbox = None
        self.savefig = None


_global_config = AppConfig()


def _runapp(app_config):
    app = BaseApp(app_config)
    app.run()


def show():
    # proc = Process(target=_runapp, args=(_global_config,))
    # proc.start()
    _runapp(_global_config)


def savefig(fname):
    _global_config.savefig = fname
    # proc = Process(target=_runapp, args=(_global_config,))
    # proc.start()
    _runapp(_global_config)


def scatter(data, color=None, point_size=6, f_tooltip=None):
    _global_config.layers.append(ScatterLayer(data, color=color, point_size=point_size, f_tooltip=f_tooltip))


def hist(data, cmap='autumn', alpha=220, binsize=16, show_tooltip=False, vmin=1, f_group=None):
    _global_config.layers.append(HistogramLayer(data, cmap=cmap, alpha=alpha, binsize=binsize, show_tooltip=show_tooltip, vmin=vmin, f_group=f_group))


def graph(data, src_lat, src_lon, dest_lat, dest_lon, **kwargs):
    _global_config.layers.append(GraphLayer(data, src_lat, src_lon, dest_lat, dest_lon, **kwargs))


def clusters(data):
    _global_config.layers.append(ClusterLayer(data))


def clear():
    _global_config.layers = []


# def tiles_provider(tiles_provider):
#     _app.map_layer.tiles_provider = tiles_provider


def add_layer(layer):
    _global_config.layers.append(layer)


def shapefiles(fname, **kwargs):
    _global_config.layers.append(PolyLayer(fname, **kwargs))


def voronoi(data, **kwargs):
    _global_config.layers.append(VoronoiLayer(data, **kwargs))


def markers(data, marker, **kwargs):
    _global_config.layers.append(MarkersLayer(data, marker, **kwargs))


def set_bbox(bbox):
    _global_config.bbox = bbox
