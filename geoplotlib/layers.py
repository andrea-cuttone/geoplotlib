from collections import defaultdict
from threading import Thread
import pyglet
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN
import numpy as np
import colors
from geoplotlib.core import BatchPainter, SCREEN_W, SCREEN_H, TILE_SIZE
from geoplotlib.utils import epoch_to_str


class ScatterLayer():

    def __init__(self, data, color=None, point_size=6, f_tooltip=None):
        self.data = data

        if color is None:
            color = [255,0,0]
        self.color = color
        self.point_size = point_size
        self.f_tooltip = f_tooltip

        self.hotspots = HotspotManager()


    def invalidate(self, proj):
        self.painter = BatchPainter()
        x, y = proj.lonlat_to_screen(self.data['lon'], self.data['lat'])
        if self.f_tooltip:
            for i in range(0, len(x)):
                self.hotspots.add_rect(x[i] - self.point_size, y[i] - self.point_size,
                                       2*self.point_size, 2*self.point_size,
                                       self.f_tooltip(self.data, i))
        self.painter.set_color(self.color)
        self.painter.points(x, y, 2*self.point_size, False)


    def draw(self, mouse_x, mouse_y, ui_manager):
        self.painter.batch_draw()
        picked = self.hotspots.pick(mouse_x, mouse_y)
        if picked:
            ui_manager.tooltip(picked)


class HotspotManager():


    def __init__(self):
        self.rects = {}


    def add_rect(self, x, y, w, h, value):
        self.rects[tuple(map(int, [x, y, w, h]))] = value


    def pick(self, mouse_x, mouse_y):
        for (x, y, w, h) in self.rects.keys():
            if (x <= mouse_x <= x + w) and (y <= mouse_y <= y + h):
                return self.rects[(x, y, w, h)]
        return None

    def __str__(self):
        return str(self.rects)


class HistogramLayer():

    def __init__(self, data):
        self.data = data


    def invalidate(self, proj):
        self.painter = BatchPainter()
        x, y = proj.lonlat_to_screen(self.data['lon'], self.data['lat'])
        bin_size = 16
        ix = x // bin_size
        iy = y // bin_size
        frequencies = defaultdict(int)
        for k in zip(ix, iy):
            frequencies[k] += 1

        self.hotspot = HotspotManager()

        vmax = max(frequencies.values())
        if vmax > 1:
            cmap = colors.create_log_cmap(vmax, 'coolwarm', alpha=200)
            for (ix, iy), value in frequencies.items():
                self.painter.set_color(cmap(value))
                self.painter.rect(ix * bin_size, iy * bin_size, (ix+1)*bin_size, (iy+1)*bin_size)
                self.hotspot.add_rect(ix * bin_size, iy * bin_size, bin_size, bin_size, 'Value: %d' % value)


    def draw(self, mouse_x, mouse_y, ui_manager):
        self.painter.batch_draw()
        picked = self.hotspot.pick(mouse_x, mouse_y)
        if picked:
            ui_manager.tooltip(picked)


class GraphLayer():

    def __init__(self, lat0, lon0, lat1, lon1, **kwargs):
        self.lon = np.vstack((lon0, lon1)).T.flatten()
        self.lat = np.vstack((lat0, lat1)).T.flatten()
        self.linewidth = kwargs.get('linewidth', 1.0)
        self.color = kwargs.get('color', [255,0,0,255])


    def invalidate(self, proj):
        self.painter = BatchPainter()
        x, y = proj.lonlat_to_screen(self.lon, self.lat)
        self.painter.set_color(self.color)
        self.painter.lines(x, y, self.linewidth)


    def draw(self, mouse_x, mouse_y, ui_manager):
        self.painter.batch_draw()


class KDELayer():

    def __init__(self, data):
        self.data = data
        self.invalidate()


    def invalidate(self, proj):
        self.painter = BatchPainter()

        x, y = proj.lonlat_to_screen(self.data['lon'], self.data['lat'])
        values = np.vstack((x,y)).T
        L = 16
        print proj.zoom
        x_flat = np.arange(x.min(), x.max(), L)
        y_flat = np.arange(y.min(), y.max(), L)
        print x_flat.shape, y_flat.shape
        x,y = np.meshgrid(x_flat,y_flat)
        grid_coords = np.append(x.reshape(-1,1),y.reshape(-1,1),axis=1)

        kde = gaussian_kde(values.T, bw_method=.05)
        print 'kde done'
        z = kde(grid_coords.T)
        z = z.reshape(len(y_flat), len(x_flat))

        vmax = z.max()
        cmap = colors.create_linear_cmap(1.0, 'coolwarm', alpha=200)

        for i in xrange(len(x_flat) - 1):
            for j in xrange(len(y_flat) - 1):
                if z[j,i] > 0:
                    self.painter.set_color(cmap((z[j,i] / vmax)**.5))
                    self.painter.rect(x_flat[i], y_flat[j], x_flat[i+1], y_flat[j+1])

# TODO: move in
class WorkerThread(Thread):

    def __init__(self, proj, data):
        Thread.__init__(self)
        self.proj = proj
        self.data = data


    def run(self):
        print 'running'
        x, y = self.proj.lonlat_to_screen(self.data['lon'], self.data['lat'])
        self.X = np.vstack((x,y)).T
        self.dbscan = DBSCAN(eps=50, min_samples=2)
        self.dbscan.fit(self.X)
        print 'done'


class ClusterLayer():

    def __init__(self, data):
        self.data = data


    def invalidate(self, proj):
        print 'invalidating'
        self.painter = BatchPainter()
        x, y = proj.lonlat_to_screen(self.data['lon'], self.data['lat'])
        self.painter.points(x, y)
        # TODO: synch arguments?
        self.worker = WorkerThread(proj, self.data)
        self.worker.start()


    def draw(self, mouse_x, mouse_y, ui_manager):
        if self.worker is not None and self.worker.is_alive() == False:
            labels = self.worker.dbscan.labels_
            cmap = colors.create_set_cmap(labels, 'jet')
            for l in set(labels):
                if l != -1:
                    points = self.worker.X[labels == l]
                    self.painter.set_color(cmap[l])
                    self.painter.points(points[:,0], points[:,1])
            self.worker = None
        self.painter.batch_draw()
