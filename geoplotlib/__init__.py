import traceback
from geoplotlib.core import BaseApp


class AppConfig:

    def __init__(self):
        self.reset()


    def reset(self):
        self.layers = []
        self.bbox = None
        self.savefig = None
        self.tiles_provider = 'mapquest'
        self.smoothing = False
        self.map_alpha = 196


_global_config = AppConfig()


def _runapp(app_config):
    app = BaseApp(app_config)
    try:
        app.start()
    except:
        traceback.print_exc()
    finally:
        app.close()
        _global_config.reset()


def show():
    _runapp(_global_config)


def savefig(fname):
    _global_config.savefig = fname
    _runapp(_global_config)


def scatter(data, color=None, point_size=3, f_tooltip=None):
    from geoplotlib.layers import ScatterLayer
    _global_config.layers.append(ScatterLayer(data, color=color, point_size=point_size, f_tooltip=f_tooltip))


def hist(data, cmap='hot', alpha=220, colorscale='sqrt', binsize=16, show_tooltip=False,
         scalemin=0, scalemax=None, f_group=None, binscaling=False):
    from geoplotlib.layers import HistogramLayer
    _global_config.layers.append(HistogramLayer(data, cmap=cmap, alpha=alpha, binsize=binsize,
            show_tooltip=show_tooltip, scalemin=scalemin, scalemax=scalemax, f_group=f_group,
            colorscale=colorscale, binscaling=binscaling))


def graph(data, src_lat, src_lon, dest_lat, dest_lon, **kwargs):
    from geoplotlib.layers import GraphLayer
    _global_config.layers.append(GraphLayer(data, src_lat, src_lon, dest_lat, dest_lon, **kwargs))


def shapefiles(fname, **kwargs):
    from geoplotlib.layers import PolyLayer
    _global_config.layers.append(PolyLayer(fname, **kwargs))


def voronoi(data, **kwargs):
    from geoplotlib.layers import VoronoiLayer
    _global_config.layers.append(VoronoiLayer(data, **kwargs))


def delaunay(data, **kwargs):
    from geoplotlib.layers import DelaunayLayer
    _global_config.layers.append(DelaunayLayer(data, **kwargs))


def kde(data, bw, **kwargs):
    from geoplotlib.layers import KDELayer
    _global_config.layers.append(KDELayer(data, bw, **kwargs))


def markers(data, marker, **kwargs):
    from geoplotlib.layers import MarkersLayer
    _global_config.layers.append(MarkersLayer(data, marker, **kwargs))


def clear():
    _global_config.layers = []


def tiles_provider(tiles_provider):
    _global_config.tiles_provider = tiles_provider


def add_layer(layer):
    _global_config.layers.append(layer)


def set_bbox(bbox):
    _global_config.bbox = bbox


def set_smoothing(smoothing):
    _global_config.smoothing = smoothing


def set_map_alpha(alpha):
    if alpha < 0 or alpha > 255:
        raise Exception('invalid alpha '  + str(alpha))
    _global_config.map_alpha = alpha
