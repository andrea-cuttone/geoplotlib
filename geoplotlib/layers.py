from collections import defaultdict
from math import log10, log
from threading import Thread
import threading
import math
import pyglet
import numpy as np
import colors
from geoplotlib.core import BatchPainter
from geoplotlib.utils import BoundingBox
import Queue
from inspect import isfunction
import json
from core import FONT_NAME


class HotspotManager():


    def __init__(self):
        self.rects = []
        self.poly = []


    # adapted from:
    # http://stackoverflow.com/questions/16625507/python-checking-if-point-is-inside-a-polygon
    @staticmethod
    def point_in_poly(x, y, bbox, poly):
        left, top, right, bottom = bbox
        if x < left or x > right or y < top or y > bottom:
            return False

        n = len(poly)
        is_inside = False

        x1,y1 = poly[0]
        for i in range(n+1):
            x2,y2 = poly[i % n]
            if y > min(y1,y2):
                if y <= max(y1,y2):
                    if x <= max(x1,x2):
                        if y1 != y2:
                            xints = (y-y1)*(x2-x1)/(y2-y1)+x1
                        if x1 == x2 or x <= xints:
                            is_inside = not is_inside
            x1,y1 = x2,y2

        return is_inside


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
    """
    Base class for layers
    """

    def invalidate(self, proj):
        """
        This method is called each time layers need to be redrawn, i.e. on zoom.
        Typically in this method a BatchPainter is instantiated and all the rendering is performed

        :param proj: the current Projector object
        """
        pass


    def draw(self, proj, mouse_x, mouse_y, ui_manager):
        """
        This method is called at every frame, and typically executes BatchPainter.batch_draw()

        :param proj: the current Projector object
        :param mouse_x: mouse x
        :param mouse_y: mouse y
        :param ui_manager: the current UiManager
        """
        pass


    def bbox(self):
        """
        Return the bounding box for this layer
        """
        return BoundingBox.WORLD


    def on_key_release(self, key, modifiers):
        """
        Override this method for custom handling of keystrokes

        :param key: the key that has been released
        :param modifiers: the key modifiers
        :return: True if the layer needs to call invalidate
        """
        return False


class DotDensityLayer(BaseLayer):


    def __init__(self, data, color=None, point_size=2, f_tooltip=None):
        """Create a dot density map

        :param data: data access object
        :param color: color
        :param point_size: point size
        :param f_tooltip: function to return a tooltip string for a point
        """
        self.data = data
        self.color = color
        if self.color is None:
            self.color = [255,0,0]
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


    def draw(self, proj, mouse_x, mouse_y, ui_manager):
        self.painter.batch_draw()
        picked = self.hotspots.pick(mouse_x, mouse_y)
        if picked:
            ui_manager.tooltip(picked)


    def bbox(self):
        return BoundingBox.from_points(lons=self.data['lon'], lats=self.data['lat'])


class HistogramLayer(BaseLayer):

    def __init__(self, data, cmap='hot', alpha=220, colorscale='sqrt', binsize=16, 
                 show_tooltip=False, scalemin=0, scalemax=None, f_group=None, show_colorbar=True):
        """Create a 2D histogram

        :param data: data access object
        :param cmap: colormap name
        :param alpha: color alpha
        :param colorscale: scaling [lin, log, sqrt]
        :param binsize: size of the hist bins
        :param show_tooltip: if True, will show the value of bins on mouseover
        :param scalemin: min value for displaying a bin
        :param scalemax: max value for a bin
        :param f_group: function to apply to samples in the same bin. Default is to count
        :param show_colorbar: show colorbar
        :return:
        """
        self.data = data
        self.cmap = colors.ColorMap(cmap, alpha=alpha)
        self.binsize = binsize
        self.show_tooltip = show_tooltip
        self.scalemin = scalemin
        self.scalemax = scalemax
        self.colorscale = colorscale
        self.f_group = f_group
        if self.f_group is None:
            self.f_group = lambda grp: len(grp)
        self.show_colorbar = show_colorbar


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

        if self.scalemax:
            self.vmax = self.scalemax
        else:
            self.vmax = max(results.values()) if len(results) > 0 else 0

        if self.vmax >= 1:
            for (ix, iy), value in results.items():
                if value > self.scalemin:
                    self.painter.set_color(self.cmap.to_color(value, self.vmax, self.colorscale))
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
        if self.show_colorbar:
            ui_manager.add_colorbar(self.cmap, self.vmax, self.colorscale)


    def bbox(self):
        return BoundingBox.from_points(lons=self.data['lon'], lats=self.data['lat'])


class GraphLayer(BaseLayer):

    def __init__(self, data, src_lat, src_lon, dest_lat, dest_lon, linewidth=1, alpha=220, color='hot'):
        """Create a graph drawing a line between each pair of (src_lat, src_lon) and (dest_lat, dest_lon)

        :param data: data access object
        :param src_lat: field name of source latitude
        :param src_lon: field name of source longitude
        :param dest_lat: field name of destination latitude
        :param dest_lon: field name of destination longitude
        :param linewidth: line width
        :param alpha: color alpha
        :param color: color or colormap
        """
        self.data = data
        self.src_lon = src_lon
        self.src_lat = src_lat
        self.dest_lon = dest_lon
        self.dest_lat = dest_lat

        self.linewidth = linewidth
        alpha = alpha
        self.color = color
        if type(self.color) == str:
            self.cmap = colors.ColorMap(self.color, alpha)


    def invalidate(self, proj):
        self.painter = BatchPainter()
        x0, y0 = proj.lonlat_to_screen(self.data[self.src_lon], self.data[self.src_lat])
        x1, y1 = proj.lonlat_to_screen(self.data[self.dest_lon], self.data[self.dest_lat])

        if type(self.color) == list:
            self.painter.set_color(self.color)
            self.painter.lines(x0, y0, x1, y1, width=self.linewidth)
        else:
            manhattan = np.abs(x0-x1) + np.abs(y0-y1)
            vmax = manhattan.max()
            distances = np.logspace(0, log10(manhattan.max()), 20)
            for i in range(len(distances)-1, 1, -1):
                mask = (manhattan > distances[i-1]) & (manhattan <= distances[i])
                self.painter.set_color(self.cmap.to_color(distances[i], vmax, 'log'))
                self.painter.lines(x0[mask], y0[mask], x1[mask], y1[mask], width=self.linewidth)


    def draw(self, proj, mouse_x, mouse_y, ui_manager):
        self.painter.batch_draw()


    def bbox(self):
        return BoundingBox.from_points(lons=np.hstack([self.data[self.src_lon], self.data[self.dest_lon]]),
                                       lats=np.hstack([self.data[self.src_lat], self.data[self.dest_lat]]))


class ShapefileLayer(BaseLayer):

    def __init__(self, fname, f_tooltip=None, color=None, linewidth=3, shape_type='full'):
        """
        Loads and draws shapefiles

        :param fname: full path to the shapefile
        :param f_tooltip: function to generate a tooltip on mouseover
        :param color: color
        :param linewidth: line width
        :param shape_type: either full or bbox
        """
        if color is None:
            color = [255, 0, 0]
        self.color = color
        self.linewidth = linewidth
        self.f_tooltip = f_tooltip
        self.shape_type = shape_type

        try:
            import shapefile
        except:
            raise Exception('ShapefileLayer requires pyshp')

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

    def __init__(self, data, line_color=None, line_width=2, cmap=None, max_lenght=100):
        """
        Draw a delaunay triangulation of the points

        :param data: data access object
        :param line_color: line color
        :param line_width: line width
        :param cmap: color map
        :param max_lenght: scaling constant for coloring the edges
        """
        self.data = data

        if cmap is None and line_color is None:
            raise Exception('need either cmap or line_color')

        if cmap is not None:
            cmap = colors.ColorMap(cmap, alpha=196)

        self.cmap = cmap
        self.line_color = line_color
        self.line_width = line_width
        self.max_lenght = max_lenght


    @staticmethod
    def _get_area(p):
        x1, y1, x2, y2, x3, y3 = p
        return 0.5*(x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2))



    def invalidate(self, proj):
        try:
            from scipy.spatial.qhull import Delaunay
        except ImportError:
            print('DelaunayLayer needs scipy >= 0.12')
            raise

        self.painter = BatchPainter()
        x, y = proj.lonlat_to_screen(self.data['lon'], self.data['lat'])
        points = list(set(zip(x,y)))
        dela = Delaunay(points)

        edges = set()
        for tria in dela.vertices:
            edges.add((tria[0], tria[1]))
            edges.add((tria[1], tria[2]))
            edges.add((tria[2], tria[0]))

        allx0 = []
        ally0 = []
        allx1 = []
        ally1 = []
        colors = []

        for a, b in edges:
            x0, y0 = dela.points[a]
            x1, y1 = dela.points[b]

            allx0.append(x0)
            ally0.append(y0)
            allx1.append(x1)
            ally1.append(y1)

            if self.line_color:
                colors.append(self.line_color)
                colors.append(self.line_color)
            elif self.cmap:
                l = math.sqrt((x0 - x1)**2+(y0 - y1)**2)
                c = self.cmap.to_color(l, self.max_lenght, 'log')
                colors.append(c)
                colors.append(c)

        self.painter.lines(allx0, ally0, allx1, ally1, colors, width=self.line_width)


    def draw(self, proj, mouse_x, mouse_y, ui_manager):
        self.painter.batch_draw()


    def bbox(self):
        return BoundingBox.from_points(lons=self.data['lon'], lats=self.data['lat'])


class VoronoiLayer(BaseLayer):

    def __init__(self, data, line_color=None, line_width=2, f_tooltip=None, cmap=None, max_area=1e4, alpha=220):
        """
        Draw the voronoi tesselation of the points from data

        :param data: data access object
        :param line_color: line color
        :param line_width: line width
        :param f_tooltip: function to generate a tooltip on mouseover
        :param cmap: color map
        :param max_area: scaling constant to determine the color of the voronoi areas
        :param alpha: color alpha
        :return:
        """
        self.data = data

        if cmap is None and line_color is None:
            raise Exception('need either cmap or line_color')

        if cmap is not None:
            cmap = colors.ColorMap(cmap, alpha=alpha, levels=10)

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


    # Area of a polygon: http://www.mathopenref.com/coordpolygonarea.html
    @staticmethod
    def _get_area(p):
        return 0.5 * abs(sum(x0*y1 - x1*y0
                             for ((x0, y0), (x1, y1)) in zip(p, p[1:] + [p[0]])))


    def invalidate(self, proj):
        try:
            from scipy.spatial.qhull import Voronoi
        except ImportError:
            print('VoronoiLayer needs scipy >= 0.12')
            raise

        x, y = proj.lonlat_to_screen(self.data['lon'], self.data['lat'])
        points = zip(x,y)
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
                area = max(area, 1)
                self.painter.set_color(self.cmap.to_color(area, self.max_area, 'log'))
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

    def __init__(self, data, marker, f_tooltip=None, marker_preferred_size=32):
        """
        Draw markers

        :param data: data access object
        :param marker: full filename of the marker image
        :param f_tooltip: function to generate a tooltip on mouseover
        :param marker_preferred_size: size in pixel for the marker images
        """
        self.data = data
        self.f_tooltip = f_tooltip
        self.marker_preferred_size = float(marker_preferred_size)
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


class KDELayer(BaseLayer):

    def __init__(self, values, bw, cmap='hot', method='hist', scaling='sqrt', alpha=220,
                 cut_below=None, clip_above=None, binsize=1, cmap_levels=10, show_colorbar=False):
        """
        Kernel density estimation visualization

        :param data: data access object
        :param bw: kernel bandwidth (in screen coordinates)
        :param cmap: colormap
        :param method: if kde use KDEMultivariate from statsmodel, which provides a more accurate but much slower estimation.
            If hist, estimates density applying gaussian smoothing on a 2D histogram, which is much faster but less accurate
        :param scaling: colorscale, lin log or sqrt
        :param alpha: color alpha
        :param cut_below: densities below cut_below are not drawn
        :param clip_above: defines the max value for the colorscale
        :param binsize: size of the bins for hist estimator
        :param cmap_levels: discretize colors into cmap_levels
        :param show_colorbar: show colorbar
        """
        self.values = values
        self.bw = bw
        self.cmap = colors.ColorMap(cmap, alpha=alpha, levels=cmap_levels)
        self.method = method
        self.scaling = scaling
        self.cut_below = cut_below
        self.clip_above = clip_above
        self.binsize = binsize
        self.show_colorbar = show_colorbar


    def _get_grid(self, proj):
        west, north = proj.lonlat_to_screen([proj.bbox().west], [proj.bbox().north])
        east, south = proj.lonlat_to_screen([proj.bbox().east], [proj.bbox().south])
        xgrid = np.arange(west, east, self.binsize)
        ygrid = np.arange(south, north, self.binsize)
        return xgrid, ygrid


    def invalidate(self, proj):
        self.painter = BatchPainter()
        xv, yv = proj.lonlat_to_screen(self.values['lon'], self.values['lat'])

        rects_vertices = []
        rects_colors = []

        if self.method == 'kde':
            try:
                import statsmodels.api as sm
            except:
                raise Exception('KDE requires statsmodel')

            kde_res = sm.nonparametric.KDEMultivariate(data=[xv, yv], var_type='cc', bw=self.bw)
            xgrid, ygrid = self._get_grid(proj)
            xmesh, ymesh = np.meshgrid(xgrid,ygrid)
            grid_coords = np.append(xmesh.reshape(-1,1), ymesh.reshape(-1,1),axis=1)
            z = kde_res.pdf(grid_coords.T)
            z = z.reshape(len(ygrid), len(xgrid))

            # np.save('z.npy', z)
            # z = np.load('z.npy')

            print('smallest non-zero density:', z[z > 0][0])
            print('max density:', z.max())

            if self.cut_below is None:
                zmin = z[z > 0][0]
            else:
                zmin = self.cut_below

            if self.clip_above is None:
                zmax = z.max()
            else:
                zmax = self.clip_above

            for ix in range(len(xgrid)-1):
                for iy in range(len(ygrid)-1):
                    if z[iy, ix] > zmin:
                        rects_vertices.append((xgrid[ix], ygrid[iy], xgrid[ix+1], ygrid[iy+1]))
                        rects_colors.append(self.cmap.to_color(z[iy, ix], zmax, self.scaling))
        elif self.method == 'hist':
            try:
                from scipy.ndimage import gaussian_filter
            except:
                raise Exception('KDE requires scipy')

            xgrid, ygrid = self._get_grid(proj)
            H, _, _ = np.histogram2d(yv, xv, bins=(ygrid, xgrid))
            
            if H.sum() == 0:
                print('no data in current view')
                return

            H = gaussian_filter(H, sigma=self.bw)
            print('smallest non-zero count', H[H > 0][0])
            print('max count:', H.max())

            if self.cut_below is None:
                Hmin = H[H > 0][0]
            else:
                Hmin = self.cut_below

            if self.clip_above is None:
                self.Hmax = H.max()
            else:
                self.Hmax = self.clip_above

            if self.scaling == 'ranking':
                from statsmodels.distributions.empirical_distribution import ECDF
                ecdf = ECDF(H.flatten())

            for ix in range(len(xgrid)-2):
                for iy in range(len(ygrid)-2):
                    if H[iy, ix] > Hmin:
                        rects_vertices.append((xgrid[ix], ygrid[iy], xgrid[ix+1], ygrid[iy+1]))
                        if self.scaling == 'ranking':
                            rects_colors.append(self.cmap.to_color(ecdf(H[iy, ix]) - ecdf(Hmin), 1 - ecdf(Hmin), 'lin'))
                        else:
                            rects_colors.append(self.cmap.to_color(H[iy, ix], self.Hmax, self.scaling))
        else:
            raise Exception('method not supported')

        self.painter.batch_rects(rects_vertices, rects_colors)


    def draw(self, proj, mouse_x, mouse_y, ui_manager):
        self.painter.batch_draw()
        if self.show_colorbar:
            ui_manager.add_colorbar(self.cmap, self.Hmax, self.scaling)


class ConvexHullLayer(BaseLayer):

    def __init__(self, data, col, fill=True, point_size=4):
        """
        Convex hull for a set of points

        :param data: points
        :param col: color
        :param fill: whether to fill the convexhull polygon or not
        :param point_size: size of the points on the convexhull. Points are not rendered if None
        """
        self.data = data
        self.col = col
        self.fill = fill
        self.point_size=point_size


    def invalidate(self, proj):
        self.painter = BatchPainter()
        self.painter.set_color(self.col)
        x, y = proj.lonlat_to_screen(self.data['lon'], self.data['lat'])
        if len(x) >= 3:
            self.painter.convexhull(x, y, self.fill)
        else:
            self.painter.linestrip(x, y)

        if self.point_size > 0:
            self.painter.points(x, y, self.point_size)


    def draw(self, proj, mouse_x, mouse_y, ui_manager):
        self.painter.batch_draw()


class GridLayer(BaseLayer):

    def __init__(self, lon_edges, lat_edges, values, cmap, alpha=255, vmin=None, vmax=None, levels=10, 
            colormap_scale='lin', show_colorbar=True):
        """
        Values over a uniform grid
        
        :param lon_edges: longitude edges
        :param lat_edges: latitude edges
        :param values: matrix representing values on the grid 
        :param cmap: colormap name
        :param alpha: color alpha
        :param vmin: minimum value for the colormap
        :param vmax: maximum value for the colormap
        :param levels: number of levels for the colormap
        :param colormap_scale: colormap scale
        :param show_colorbar: show the colorbar in the UI
        """
        self.lon_edges = lon_edges
        self.lat_edges = lat_edges
        self.values = values
        self.cmap = colors.ColorMap(cmap, alpha=alpha, levels=levels)
        self.colormap_scale = colormap_scale
        self.show_colorbar = show_colorbar

        if vmin:
            self.vmin = vmin
        else:
            self.vmin = 0

        if vmax:
            self.vmax = vmax
        else:
            self.vmax = self.values[~np.isnan(self.values)].max()


    def invalidate(self, proj):
        self.painter = BatchPainter()
        xv, yv = proj.lonlat_to_screen(self.lon_edges, self.lat_edges)

        rects = []
        cols = []
        for ix in range(len(xv)-1):
            for iy in range(len(yv)-1):
                d = self.values[iy, ix]
                if d > self.vmin:
                    rects.append((xv[ix], yv[iy], xv[ix+1], yv[iy+1]))
                    cols.append(self.cmap.to_color(d, self.vmax, self.colormap_scale))

        self.painter.batch_rects(rects, cols)


    def draw(self, proj, mouse_x, mouse_y, ui_manager):
        self.painter.batch_draw()
        if self.show_colorbar:
            ui_manager.add_colorbar(self.cmap, self.vmax, self.colormap_scale)


    def bbox(self):
        return BoundingBox(north=self.lat_edges[-1], south=self.lat_edges[0], 
                           west=self.lon_edges[0], east=self.lon_edges[-1])


class GeoJSONLayer(BaseLayer):

    def __init__(self, geojson_or_fname, color='b', linewidth=1, fill=False, f_tooltip=None):
        self.color = color
        self.linewidth = linewidth
        self.fill = fill
        self.f_tooltip = f_tooltip

        if type(geojson_or_fname) == str:
            with open(geojson_or_fname) as fin:
                self.data = json.load(fin)
        elif type(geojson_or_fname) == dict:
            self.data = geojson_or_fname
        else:
            raise Exception('must provide either dict or filename')

        self.boundingbox = None
        
        for feature in self.data['features']:
            if feature['geometry']['type'] == 'Polygon':
                for poly in feature['geometry']['coordinates']: 
                    poly = np.array(poly)
                    self.__update_bbox(poly[:,0], poly[:,1])
            elif feature['geometry']['type'] == 'MultiPolygon':
                for multipoly in feature['geometry']['coordinates']:
                    for poly in multipoly: 
                        poly = np.array(poly)
                        self.__update_bbox(poly[:,0], poly[:,1])
            elif feature['geometry']['type'] == 'Point':
                lon,lat = feature['geometry']['coordinates']
                self.__update_bbox(np.array([lon]), np.array([lat]))
            elif feature['geometry']['type'] == 'LineString':
                line = np.array(feature['geometry']['coordinates'])
                self.__update_bbox(line[:,0], line[:,1])


    def __update_bbox(self, lon, lat):
        if self.boundingbox is None:
            self.boundingbox = BoundingBox(north=lat.max(), south=lat.min(), west=lon.min(), east=lon.max())
        else:
            self.boundingbox = BoundingBox(
                                    north=max(self.boundingbox.north, lat.max()),
                                    south=min(self.boundingbox.south, lat.min()),
                                    west=min(self.boundingbox.west, lon.min()),
                                    east=max(self.boundingbox.east, lon.max()))


    def invalidate(self, proj):
        self.painter = BatchPainter()
        self.hotspots = HotspotManager()

        for feature in self.data['features']:
            if isfunction(self.color):
                self.painter.set_color(self.color(feature['properties']))
            else:
                self.painter.set_color(self.color)

            if feature['geometry']['type'] == 'Polygon':
                for poly in feature['geometry']['coordinates']: 
                    poly = np.array(poly)
                    x, y = proj.lonlat_to_screen(poly[:,0], poly[:,1])
                    if self.fill:
                        self.painter.poly(x, y)
                    else:
                        self.painter.linestrip(x, y, self.linewidth, closed=True)

                    if self.f_tooltip:
                        self.hotspots.add_poly(x, y, self.f_tooltip(feature['properties']))

            elif feature['geometry']['type'] == 'MultiPolygon':
                for multipoly in feature['geometry']['coordinates']:
                    for poly in multipoly: 
                        poly = np.array(poly)
                        x, y = proj.lonlat_to_screen(poly[:,0], poly[:,1])
                        if self.fill:
                            self.painter.poly(x, y)
                        else:
                            self.painter.linestrip(x, y, self.linewidth, closed=True)

                        if self.f_tooltip:
                            self.hotspots.add_poly(x, y, self.f_tooltip(feature['properties']))
            
            elif feature['geometry']['type'] == 'Point':
                lon,lat = feature['geometry']['coordinates']
                x, y = proj.lonlat_to_screen(np.array([lon]), np.array([lat]))
                self.painter.points(x, y)
            elif feature['geometry']['type'] == 'LineString':
                line = np.array(feature['geometry']['coordinates'])
                x, y = proj.lonlat_to_screen(line[:,0], line[:,1])
                self.painter.linestrip(x, y, self.linewidth, closed=False)
            else:
                print('unknow geometry %s' % feature['geometry']['type'])


    def draw(self, proj, mouse_x, mouse_y, ui_manager):
        self.painter.batch_draw()
        picked = self.hotspots.pick(mouse_x, mouse_y)
        if picked:
            ui_manager.tooltip(picked)


    def bbox(self):
        if self.boundingbox:
            return self.boundingbox
        else:
            return BoundingBox.WORLD


class LabelsLayer(BaseLayer):


    def __init__(self, data, label_column, color=None, font_name=FONT_NAME, font_size=14, anchor_x='left', anchor_y='top'):
        """Create a layer with a text label for each sample

        :param data: data access object
        :param label_column: column in the data access object where the labels text is stored
        :param color: color
        :param font_name: font name
        :param font_size: font size
        :param anchor_x: anchor x
        :param anchor_y: anchor y
        """
        self.data = data
        self.label_column = label_column
        self.color = color
        self.font_name = font_name
        self.font_size = font_size
        self.anchor_x = anchor_x
        self.anchor_y = anchor_y
        if self.color is None:
            self.color = [255,0,0]


    def invalidate(self, proj):
        self.painter = BatchPainter()
        x, y = proj.lonlat_to_screen(self.data['lon'], self.data['lat'])
        self.painter.set_color(self.color)
        self.painter.labels(x, y, self.data[self.label_column], 
                                    font_name=self.font_name,
                                    font_size=self.font_size,
                                    anchor_x=self.anchor_x, 
                                    anchor_y=self.anchor_y)


    def draw(self, proj, mouse_x, mouse_y, ui_manager):
        self.painter.batch_draw()
        

    def bbox(self):
        return BoundingBox.from_points(lons=self.data['lon'], lats=self.data['lat'])
