import Queue
from collections import defaultdict
from threading import Thread
import threading
import pyglet
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN
import numpy as np
import colors
from geoplotlib.core import BatchPainter, SCREEN_W, SCREEN_H, TILE_SIZE
from geoplotlib.utils import epoch_to_str, parse_raw_str
import shapefile

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
                record = {k: self.data[k][i] for k in self.data.keys()}
                self.hotspots.add_rect(x[i] - self.point_size, y[i] - self.point_size,
                                       2*self.point_size, 2*self.point_size,
                                       self.f_tooltip(record))
        self.painter.set_color(self.color)
        self.painter.points(x, y, 2*self.point_size, False)


    def draw(self, mouse_x, mouse_y, ui_manager):
        self.painter.batch_draw()
        picked = self.hotspots.pick(mouse_x, mouse_y)
        if picked:
            ui_manager.tooltip(picked)


class HotspotManager():


    def __init__(self):
        self.rects = []
        self.poly = []


    @staticmethod
    # adapted from:
    # http://stackoverflow.com/questions/16625507/python-checking-if-point-is-inside-a-polygon
    def point_in_poly(x, y, bbox, poly):
        left, top, right, bottom = bbox
        if x < left or x > right or y < top or y > bottom:
            return False

        n = len(poly)
        inside = False

        p1x,p1y = poly[0]
        for i in range(n+1):
            p2x,p2y = poly[i % n]
            if y > min(p1y,p2y):
                if y <= max(p1y,p2y):
                    if x <= max(p1x,p2x):
                        if p1y != p2y:
                            xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                        if p1x == p2x or x <= xints:
                            inside = not inside
            p1x,p1y = p2x,p2y

        return inside

    def add_rect(self, x, y, w, h, value):
        self.rects.append(((x, y, w, h), value))


    def add_poly(self, x, y, value):
        bbox = (x.min(), y.min(), x.max(), y.max())
        self.poly.append((zip(x,y), bbox, value))


    def pick(self, mouse_x, mouse_y):
        for (x, y, w, h), value in self.rects:
            if (x <= mouse_x <= x + w) and (y <= mouse_y <= y + h):
                return value

        for points, bbox, value in self.poly:
            if HotspotManager.point_in_poly(mouse_x, mouse_y, bbox, points):
                return value

        return None


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


class PolyLayer():

    def __init__(self, fname, f_tooltip=None, color=None, linewidth=3, shape_type='full'):
        if color is None:
            color = [255, 0, 0]
        self.color = color
        self.linewidth = linewidth
        self.f_tooltip = f_tooltip
        self.shape_type = shape_type

        self.reader = shapefile.Reader(fname)
        self.worker = None


    def invalidate(self, proj):
        self.painter = BatchPainter()
        self.hotspots = HotspotManager()
        self.painter.set_color(self.color)

        if self.worker:
            self.worker.stop()
            self.worker.join()
        self.queue = Queue.Queue()
        self.worker = ShapeLoadingThread(self.queue, self.reader, self.shape_type, proj)
        self.worker.start()


    def draw(self, mouse_x, mouse_y, ui_manager):
        self.painter.batch_draw()
        picked = self.hotspots.pick(mouse_x, mouse_y)
        if picked:
            ui_manager.tooltip(picked)


    def on_tick(self, dt, proj):
        while True:
            try:
                x, y, record = self.queue.get_nowait()
                self.painter.linestrip(x, y, self.linewidth, closed=True)
                if self.f_tooltip:
                    attr = {t[0][0]: parse_raw_str(t[1]) for t in zip(self.reader.fields[1:], record)}
                    value = self.f_tooltip(attr)
                    if self.shape_type == 'bbox':
                        self.hotspots.add_rect(x.min(), y.min(), x.max()-x.min(), y.max()-y.min(), value)
                    else:
                        self.hotspots.add_poly(x, y, value)
            except Queue.Empty:
                break


class ShapeLoadingThread(Thread):

    def __init__(self, queue, reader, shape_type, proj):
        Thread.__init__(self)

        self.queue = queue
        self.reader = reader
        self.shape_type = shape_type
        self.proj = proj
        self.stop_flag = threading.Event()

        self.counter = 0

        self.daemon = True


    def stop(self):
        self.stop_flag.set()


    def run(self):
        while (self.counter < self.reader.numRecords) and (not self.stop_flag.is_set()):
            r = self.reader.shapeRecord(self.counter)
            if self.shape_type == 'bbox':
                top, left, bottom, right =  r.shape.bbox
                vertices = np.array([top, left, top, right, bottom, right, bottom, left]).reshape(-1,2)
            else:
                vertices = np.array(r.shape.points)
            x, y = self.proj.lonlat_to_screen(vertices[:,0], vertices[:,1])
            self.queue.put((x, y, r.record))
            self.counter += 1


class VoronoiLayer():

    def __init__(self, data, line_color=None, line_width=2, draw_points=False, points_color=None, f_tooltip=None):
        self.data = data
        if line_color is None:
            line_color = [255,136,0]
        self.line_color = line_color
        self.line_width = line_width
        self.draw_points = draw_points
        if points_color is None:
            points_color = [255,0,0]
        self.points_color = points_color
        self.f_tooltip = f_tooltip


    # source: https://gist.github.com/pv/8036995
    @staticmethod
    def __voronoi_finite_polygons_2d(vor, radius=None):
        """
        Reconstruct infinite voronoi regions in a 2D diagram to finite
        regions.

        Parameters
        ----------
        vor : Voronoi
            Input diagram
        radius : float, optional
            Distance to 'points at infinity'.

        Returns
        -------
        regions : list of tuples
            Indices of vertices in each revised Voronoi regions.
        vertices : list of tuples
            Coordinates for revised Voronoi vertices. Same as coordinates
            of input vertices, with 'points at infinity' appended to the
            end.

        """

        if vor.points.shape[1] != 2:
            raise ValueError("Requires 2D input")

        new_regions = []
        new_vertices = vor.vertices.tolist()

        center = vor.points.mean(axis=0)
        if radius is None:
            radius = vor.points.ptp().max()*2

        # Construct a map containing all ridges for a given point
        all_ridges = {}
        for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
            all_ridges.setdefault(p1, []).append((p2, v1, v2))
            all_ridges.setdefault(p2, []).append((p1, v1, v2))

        # Reconstruct infinite regions
        for p1, region in enumerate(vor.point_region):
            vertices = vor.regions[region]

            if all(v >= 0 for v in vertices):
                # finite region
                new_regions.append(vertices)
                continue

            # reconstruct a non-finite region
            if p1 not in all_ridges:
                continue

            ridges = all_ridges[p1]
            new_region = [v for v in vertices if v >= 0]

            for p2, v1, v2 in ridges:
                if v2 < 0:
                    v1, v2 = v2, v1
                if v1 >= 0:
                    # finite ridge: already in the region
                    continue

                # Compute the missing endpoint of an infinite ridge

                t = vor.points[p2] - vor.points[p1] # tangent
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])  # normal

                midpoint = vor.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = vor.vertices[v2] + direction * radius

                new_region.append(len(new_vertices))
                new_vertices.append(far_point.tolist())

            # sort region counterclockwise
            vs = np.asarray([new_vertices[v] for v in new_region])
            c = vs.mean(axis=0)
            angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
            new_region = np.array(new_region)[np.argsort(angles)]

            # finish
            new_regions.append(new_region.tolist())

        return new_regions, np.asarray(new_vertices)


    def invalidate(self, proj):
        try:
            from scipy.spatial.qhull import Voronoi
        except ImportError:
            print 'VoronoiLayer needs scipy >= 0.12'
            raise

        x, y = proj.lonlat_to_screen(self.data['lon'], self.data['lat'])
        points = np.vstack((x,y)).T
        vor = Voronoi(points)

        regions, vertices = VoronoiLayer.__voronoi_finite_polygons_2d(vor)

        self.hotspots = HotspotManager()
        self.painter = BatchPainter()
        self.painter.set_color(self.line_color)
        for idx, region in enumerate(regions):
            polygon = vertices[region]
            self.painter.linestrip(polygon[:,0], polygon[:,1], width=self.line_width, closed=True)
            if self.f_tooltip:
                record = {k: self.data[k][idx] for k in self.data.keys()}
                self.hotspots.add_poly(polygon[:,0], polygon[:,1], self.f_tooltip(record))
        if self.draw_points:
            self.painter.set_color(self.points_color)
            self.painter.points(x, y, 4)


    def draw(self, mouse_x, mouse_y, ui_manager):
        self.painter.batch_draw()
        picked = self.hotspots.pick(mouse_x, mouse_y)
        if picked:
            ui_manager.tooltip(picked)
