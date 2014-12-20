from collections import defaultdict
from math import log10, log
from threading import Thread
import threading
import pyglet
import numpy as np
import colors
from geoplotlib.core import BatchPainter
from geoplotlib.utils import BoundingBox
import Queue


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


class BaseLayer():

    def invalidate(self, proj):
        pass


    def draw(self, proj, mouse_x, mouse_y, ui_manager):
        pass


    def bbox(self):
        return BoundingBox.WORLD


    def on_key_release(self, key, modifiers):
        return False


class ScatterLayer(BaseLayer):

    def __init__(self, data, **kwargs):
        self.data = data
        self.color = kwargs.get('color')
        if self.color is None:
            self.color = [255,0,0]
        self.point_size = kwargs.get('point_size')
        self.f_tooltip = kwargs.get('f_tooltip')

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


    def draw(self, proj, mouse_x, mouse_y, ui_manager):
        self.painter.batch_draw()
        picked = self.hotspots.pick(mouse_x, mouse_y)
        if picked:
            ui_manager.tooltip(picked)


    def bbox(self):
        return BoundingBox.from_points(lons=self.data['lon'], lats=self.data['lat'])


class HistogramLayer(BaseLayer):

    def __init__(self, data, **kwargs):
        self.data = data
        self.alpha = kwargs.get('alpha')
        self.cmap = kwargs.get('cmap')
        self.binsize = kwargs.get('binsize')
        self.show_tooltip = kwargs.get('show_tooltip')
        self.vmin = kwargs.get('vmin')
        self.logscale = kwargs.get('logscale')
        self.f_group = kwargs.get('f_group', None)
        self.binscaling = kwargs.get('binscaling', None)
        if self.f_group is None:
            self.f_group = lambda grp: len(grp)


    def invalidate(self, proj):
        self.painter = BatchPainter()
        x, y = proj.lonlat_to_screen(self.data['lon'], self.data['lat'])
        self.data['_xbin'] = (x / self.binsize).astype(int)
        self.data['_ybin'] = (y / self.binsize).astype(int)
        uniquevalues = set([tuple(row) for row in np.vstack([self.data['_xbin'],self.data['_ybin']]).T])
        results = {(v1,v2): self.f_group(self.data.where((self.data['_xbin'] == v1) & (self.data['_ybin'] == v2))) \
                   for v1, v2 in uniquevalues}
        del self.data['_xbin']
        del self.data['_ybin']

        self.hotspot = HotspotManager()
        cmap = colors.create_linear_cmap(self.cmap, vmin=0, vmax=1.0, alpha=self.alpha)
        vmax = max(results.values()) if len(results) > 0 else 0

        if vmax > 1:
            for (ix, iy), value in results.items():
                if self.logscale:
                    value = log(value)/log(vmax)
                else:
                    value = 1.*value/vmax
                if value >= self.vmin:
                    self.painter.set_color(cmap(value))
                    if self.binscaling:
                        l = self.binsize * value * 0.95
                        rx = ix * self.binsize + 0.5 * (1-value) * self.binsize
                        ry = iy * self.binsize + 0.5 * (1-value) * self.binsize
                    else:
                        l = self.binsize
                        rx = ix * self.binsize
                        ry = iy * self.binsize

                    self.painter.rect(rx, ry, rx+l, ry+l)
                    if self.show_tooltip:
                        self.hotspot.add_rect(rx, ry, l, l, 'Value: %d' % value)


    def draw(self, proj, mouse_x, mouse_y, ui_manager):
        self.painter.batch_draw()
        picked = self.hotspot.pick(mouse_x, mouse_y)
        if picked:
            ui_manager.tooltip(picked)

    def bbox(self):
        return BoundingBox.from_points(lons=self.data['lon'], lats=self.data['lat'])


class GraphLayer(BaseLayer):

    def __init__(self, data, src_lat, src_lon, dest_lat, dest_lon, **kwargs):
        self.data = data
        self.src_lon = src_lon
        self.src_lat = src_lat
        self.dest_lon = dest_lon
        self.dest_lat = dest_lat

        self.linewidth = kwargs.get('linewidth', 3.0)
        self.cmap = kwargs.get('cmap', 'OrRd')
        self.alpha = kwargs.get('alpha', 32)


    def invalidate(self, proj):
        self.painter = BatchPainter()
        x0, y0 = proj.lonlat_to_screen(self.data[self.src_lon], self.data[self.src_lat])
        x1, y1 = proj.lonlat_to_screen(self.data[self.dest_lon], self.data[self.dest_lat])
        manhattan = np.abs(x0-x1) + np.abs(y0-y1)
        cols = colors.create_linear_cmap(self.cmap, alpha=self.alpha)
        distances = np.logspace(0, log10(manhattan.max()), 20)
        for i in range(len(distances)-1, 1, -1):
            mask = (manhattan > distances[i-1]) & (manhattan <= distances[i])
            self.painter.set_color(cols(log(distances[i])/log(manhattan.max())))
            self.painter.lines(x0[mask], y0[mask], x1[mask], y1[mask], self.linewidth)


    def draw(self, proj, mouse_x, mouse_y, ui_manager):
        self.painter.batch_draw()


    def bbox(self):
        return BoundingBox.from_points(lons=np.hstack([self.data[self.src_lon], self.data[self.dest_lon]]),
                                       lats=np.hstack([self.data[self.src_lat], self.data[self.dest_lat]]))


class PolyLayer(BaseLayer):

    def __init__(self, fname, f_tooltip=None, color=None, linewidth=3, shape_type='full'):
        if color is None:
            color = [255, 0, 0]
        self.color = color
        self.linewidth = linewidth
        self.f_tooltip = f_tooltip
        self.shape_type = shape_type

        import shapefile
        self.reader = shapefile.Reader(fname)
        self.worker = None
        self.queue = Queue.Queue()


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


    def draw(self, proj, mouse_x, mouse_y, ui_manager):
        self.painter.batch_draw()
        picked = self.hotspots.pick(mouse_x, mouse_y)
        if picked:
            ui_manager.tooltip(picked)

        while True:
            try:
                x, y, record = self.queue.get_nowait()
                self.painter.linestrip(x, y, self.linewidth, closed=True)
                if self.f_tooltip:
                    attr = {t[0][0]: t[1] for t in zip(self.reader.fields[1:], record)}
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


class DelaunayLayer(BaseLayer):

    def __init__(self, data, line_color=None, line_width=2, cmap=None, max_area=50):
        self.data = data

        if cmap is None and line_color is None:
            raise Exception('need either cmap or line_color')

        if cmap is not None:
            cmap = colors.create_linear_cmap(cmap, alpha=196)

        self.cmap = cmap
        self.line_color = line_color
        self.line_width = line_width
        self.max_area = max_area


    @staticmethod
    def _get_area(p):
        x1, y1, x2, y2, x3, y3 = p
        return 0.5*(x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2))


    def invalidate(self, proj):
        try:
            from scipy.spatial.qhull import Delaunay
        except ImportError:
            print 'DelaunayLayer needs scipy >= 0.12'
            raise

        self.painter = BatchPainter()

        x, y = proj.lonlat_to_screen(self.data['lon'], self.data['lat'])
        points = np.vstack((x,y)).T
        dela = Delaunay(points)

        for tria in dela.vertices:
            p1 = dela.points[tria[0]]
            p2 = dela.points[tria[1]]
            p3 = dela.points[tria[2]]
            tria_x = [p1[0], p2[0], p3[0]]
            tria_y = [p1[1], p2[1], p3[1]]
            if self.line_color:
                self.painter.linestrip(tria_x, tria_y, closed=True)
            if self.cmap:
                area = DelaunayLayer._get_area(np.vstack((tria_x, tria_y)).T.flatten())
                self.painter.set_color(self.cmap(1 - min(1, np.log(area) / np.log(self.max_area))))
                self.painter.poly(tria_x, tria_y)


    def draw(self, proj, mouse_x, mouse_y, ui_manager):
        self.painter.batch_draw()


    def bbox(self):
        return BoundingBox.from_points(lons=self.data['lon'], lats=self.data['lat'])


class VoronoiLayer(BaseLayer):

    def __init__(self, data, line_color=None, line_width=2, f_tooltip=None, cmap=None, max_area=1e4):
        self.data = data

        if cmap is None and line_color is None:
            raise Exception('need either cmap or line_color')

        if cmap is not None:
            cmap = colors.create_linear_cmap(cmap, alpha=196)

        self.cmap = cmap
        self.line_color = line_color
        self.line_width = line_width
        self.f_tooltip = f_tooltip
        self.max_area = max_area


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
            radius = vor.points.ptp().max()

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


    # source: https://stackoverflow.com/questions/451426/how-do-i-calculate-the-surface-area-of-a-2d-polygon
    @staticmethod
    def _get_area(p):
        return 0.5 * abs(sum(x0*y1 - x1*y0
                             for ((x0, y0), (x1, y1)) in VoronoiLayer._segments(p)))

    # source: https://stackoverflow.com/questions/451426/how-do-i-calculate-the-surface-area-of-a-2d-polygon
    @staticmethod
    def _segments(p):
        return zip(p, p[1:] + [p[0]])


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


        for idx, region in enumerate(regions):
            polygon = vertices[region]

            if self.line_color:
                self.painter.set_color(self.line_color)
                self.painter.linestrip(polygon[:,0], polygon[:,1], width=self.line_width, closed=True)
            if self.cmap:
                area = VoronoiLayer._get_area(polygon.tolist())
                self.painter.set_color(self.cmap(1 - min(1, np.log(area) / np.log(self.max_area))))
                self.painter.poly(polygon[:,0], polygon[:,1])

            if self.f_tooltip:
                record = {k: self.data[k][idx] for k in self.data.keys()}
                self.hotspots.add_poly(polygon[:,0], polygon[:,1], self.f_tooltip(record))


    def draw(self, proj, mouse_x, mouse_y, ui_manager):
        self.painter.batch_draw()
        picked = self.hotspots.pick(mouse_x, mouse_y)
        if picked:
            ui_manager.tooltip(picked)


    def bbox(self):
        return BoundingBox.from_points(lons=self.data['lon'], lats=self.data['lat'])


class MarkersLayer(BaseLayer):

    def __init__(self, data, marker, **kwargs):
        self.data = data
        self.f_tooltip = kwargs.get('f_tooltip')
        self.marker_preferred_size = kwargs.get('marker_preferred_size', 32.)
        self.marker = pyglet.image.load(marker)
        self.marker.anchor_x = self.marker.width / 2
        self.marker.anchor_y = self.marker.height / 2
        self.scale = self.marker_preferred_size / max(self.marker.width, self.marker.height)

        self.hotspots = HotspotManager()


    def invalidate(self, proj):
        self.painter = BatchPainter()
        x, y = proj.lonlat_to_screen(self.data['lon'], self.data['lat'])

        if self.f_tooltip:
            for i in range(0, len(x)):
                record = {k: self.data[k][i] for k in self.data.keys()}
                self.hotspots.add_rect(x[i] - self.marker_preferred_size/2,
                                       y[i] - self.marker_preferred_size/2,
                                       self.marker_preferred_size,
                                       self.marker_preferred_size,
                                       self.f_tooltip(record))

        self.painter.sprites(self.marker, x, y, self.scale)


    def draw(self, proj, mouse_x, mouse_y, ui_manager):
        self.painter.batch_draw()

        picked = self.hotspots.pick(mouse_x, mouse_y)
        if picked:
            ui_manager.tooltip(picked)


    def bbox(self):
        return BoundingBox.from_points(lons=self.data['lon'], lats=self.data['lat'])
