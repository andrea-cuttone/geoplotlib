#!/usr/bin/env python
# -*- coding: utf-8 -*-

from Queue import Queue
from threading import Thread
from pyglet.window import mouse
import time
from pyglet.gl import *
import math
import numpy as np
import os
import random
import urllib2
import pyglet
from os.path import expanduser
from geoplotlib import utils
from geoplotlib.utils import BoundingBox, parse_raw_str


VERT_PER_POINT = 2
FPS = 30
TILE_SIZE = 256
MIN_ZOOM = 2
MAX_ZOOM = 20
KEYBOARD_PAN = 0.2
TOTAL_INVALIDATE_DELAY = 50
FONT_COLOR = (0,0,0,255)
FONT_NAME = 'Helvetica'
FONT_SCALING = 1./100


class UiManager:

    class Colorbar():

        def __init__(self, cmap, vmax, colormap_scale, font_size, size=.5):
            self.cmap = cmap
            self.vmax = vmax
            self.colormap_scale = colormap_scale
            self.font_size = font_size
            self.size = size


        def draw(self, painter):
            total_h = SCREEN_H*self.size
            step = total_h / self.cmap.levels
            bar_w = SCREEN_W/25

            lab = pyglet.text.Label('',
                               color=FONT_COLOR,
                               font_name=FONT_NAME,
                               font_size=int(.8*self.font_size),
                               x=SCREEN_W, y=SCREEN_H,
                               anchor_x='right', anchor_y='center')

            edges, colors = self.cmap.get_boundaries(self.vmax, self.colormap_scale)

            for i in range(self.cmap.levels+1):
                if i < self.cmap.levels:
                    painter.set_color(colors[i][:-1])
                    painter.rect(SCREEN_W-2*bar_w/2, SCREEN_H-total_h*1.5+step*i, 
                                 SCREEN_W-bar_w/2, SCREEN_H-total_h*1.5+step*(i+1))
                lab.x = SCREEN_W-2*bar_w/2*1.1
                lab.y = SCREEN_H-total_h*1.5+step*i
                if self.colormap_scale == 'log':
                    lab.text = '%.2E' % edges[i]
                else:
                    lab.text = '%d' % edges[i]
                lab.draw()


    def __init__(self):
        self.font_size = int(SCREEN_W*FONT_SCALING)
        self.padding = 2
        self.labels = {}

        self.labels['status'] = pyglet.text.Label('',
                                       color=FONT_COLOR,
                                       font_name=FONT_NAME,
                                       font_size=self.font_size,
                                       x=20, y=10,
                                       anchor_x='left', anchor_y='bottom')

        self.labels['tooltip'] = pyglet.text.Label('',
                                       color=FONT_COLOR,
                                       font_name=FONT_NAME,
                                       font_size=self.font_size,
                                       x=SCREEN_W, y=SCREEN_H,
                                       anchor_x='left', anchor_y='bottom')

        self.labels['info'] = pyglet.text.Label('',
                                       color=FONT_COLOR,
                                       font_name=FONT_NAME,
                                       font_size=self.font_size,
                                       x=SCREEN_W, y=SCREEN_H,
                                       anchor_x='right', anchor_y='top')

        self.colorbar = None


    def tooltip(self, text):
        self.labels['tooltip'].text = parse_raw_str(text)


    def status(self, text):
        self.labels['status'].text = parse_raw_str(text)


    def info(self, text):
        self.labels['info'].text = parse_raw_str(text)


    @staticmethod
    def get_label_bbox(label):

        if label.anchor_x == 'left':
            left = label.x
        elif label.anchor_x == 'right':
            left = label.x - label.content_width

        if label.anchor_y == 'bottom':
            top = label.y
        elif label.anchor_y == 'top':
            top = label.y - label.content_height

        return left, top, left + label.content_width, top + label.content_height


    def draw_label_background(self, label, painter):
        if len(label.text) > 0:
            left, top, right, bottom = UiManager.get_label_bbox(label)
            painter.rect(left - self.padding, top - self.padding, right + self.padding, bottom + self.padding)


    def draw(self, mouse_x, mouse_y):
        painter = BatchPainter()

        if self.colorbar:
            self.colorbar.draw(painter)

        painter.set_color([255,255,255])
        self.labels['tooltip'].x = mouse_x
        self.labels['tooltip'].y = mouse_y
        for l in self.labels.values():
            self.draw_label_background(l, painter)
        painter.batch_draw()
        for l in self.labels.values():
            l.draw()


    def clear(self):
        for l in self.labels.values():
            l.text = ''


    def add_colorbar(self, cmap, vmax, colormap_scale):
        self.colorbar = UiManager.Colorbar(cmap, vmax, colormap_scale, self.font_size)


class GeoplotlibApp(pyglet.window.Window):

    def __init__(self, geoplotlib_config):
        super(GeoplotlibApp, self).__init__(geoplotlib_config.screen_w, geoplotlib_config.screen_h,
                                            fullscreen=False, caption='geoplotlib')
        global SCREEN_W, SCREEN_H
        SCREEN_W = geoplotlib_config.screen_w
        SCREEN_H = geoplotlib_config.screen_h

        self.geoplotlib_config = geoplotlib_config

        self.ticks = 0
        self.ui_manager = UiManager()
        self.proj = Projector()
        self.map_layer = MapLayer(geoplotlib_config.tiles_provider, skipdl=False)

        self.scroll_delay = 0
        self.invalidate_delay = 0
        self.drag_x = self.drag_y = 0
        self.dragging = False
        self.drag_start_timestamp = 0
        self.mouse_x = self.mouse_y = 0
        self.show_map = True
        self.show_layers = True
        self.show_coordinates = False

        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_POLYGON_SMOOTH)
        # glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
        # glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        pyglet.clock.schedule_interval(self.on_update, 1. / FPS)


    def on_draw(self):
        self.clear()

        # needed to avoid diagonal artifacts on the tiles
        glDisable(GL_LINE_SMOOTH)
        glDisable(GL_POLYGON_SMOOTH)

        self.ui_manager.clear()

        if self.show_map:
            self.map_layer.draw(self.proj)

            if self.geoplotlib_config.map_alpha < 255:
                painter = BatchPainter()
                painter.set_color([0,0,0, 255 - self.geoplotlib_config.map_alpha])
                painter.rect(0,0,SCREEN_W, SCREEN_H)
                painter.batch_draw()


        if abs(self.drag_x) > 1e-3 or abs(self.drag_y) > 1e-3:
            self.drag_x *= 0.93
            self.drag_y *= 0.93

            if self.dragging == False:
                self.proj.pan(self.drag_x, self.drag_y)

        if self.scroll_delay > 0:
            self.scroll_delay -= 1

        if self.invalidate_delay > 0:
            self.invalidate_delay -= 1
        if self.invalidate_delay == 1:
            for l in self.geoplotlib_config.layers:
                l.invalidate(self.proj)
        if self.show_layers and self.invalidate_delay == 0:
            if self.geoplotlib_config.smoothing:
                glEnable(GL_LINE_SMOOTH)
                glEnable(GL_POLYGON_SMOOTH)

            glPushMatrix()
            glTranslatef(-self.proj.xtile * TILE_SIZE, self.proj.ytile * TILE_SIZE, 0)
            for l in self.geoplotlib_config.layers:
                l.draw(self.proj,
                       self.mouse_x + self.proj.xtile * TILE_SIZE,
                       SCREEN_H - self.mouse_y - self.proj.ytile * TILE_SIZE,
                       self.ui_manager)
            glPopMatrix()

            #self.ui_manager.status('T: %.1f, FPS:%d' % (self.ticks / 1000., pyglet.clock.get_fps()))
            if self.show_coordinates:
                self.ui_manager.status('%.6f %.6f' % self.proj.screen_to_latlon(self.mouse_x, SCREEN_H - self.mouse_y))

        if self.invalidate_delay == 2:
            self.ui_manager.status('rendering...')

        attribution = pyglet.text.Label(self.map_layer.attribution,
                                       color=FONT_COLOR,
                                       font_name=FONT_NAME,
                                       font_size=int(.8*SCREEN_W*FONT_SCALING),
                                       x=SCREEN_W-int(.2*SCREEN_W*FONT_SCALING), 
                                       y=int(1.2*SCREEN_W*FONT_SCALING),
                                       anchor_x='right', anchor_y='top')
        attribution.draw()
        self.ui_manager.draw(self.mouse_x, SCREEN_H - self.mouse_y)

        if self.geoplotlib_config.savefig is not None:
            GeoplotlibApp.screenshot(self.geoplotlib_config.savefig + '.png')
            pyglet.app.exit()


    def on_mouse_motion(self, x, y, dx, dy):
        self.mouse_x = x
        self.mouse_y = SCREEN_H - y


    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if buttons & mouse.LEFT:
            self.drag_start_timestamp = self.ticks
            self.drag_x = -1. * dx / TILE_SIZE
            self.drag_y = -1. * dy / TILE_SIZE
            self.proj.pan(self.drag_x, self.drag_y)
            if self.invalidate_delay > 0:
                    self.invalidate_delay = TOTAL_INVALIDATE_DELAY


    def on_mouse_release(self, x, y, buttons, modifiers):
        if buttons == mouse.LEFT:
            self.dragging = False
            if self.ticks - self.drag_start_timestamp > 200:
                self.drag_x = self.drag_y = 0


    def on_mouse_press(self, x, y, buttons, modifiers):
        if buttons == mouse.LEFT:
            if not self.dragging:
                self.dragging = True
                self.drag_start_timestamp = self.ticks
                self.drag_x = self.drag_y = 0
                if self.invalidate_delay > 0:
                    self.invalidate_delay = TOTAL_INVALIDATE_DELAY


    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        if self.scroll_delay == 0:
            if scroll_y < 0:
                self.proj.zoomin(self.mouse_x, self.mouse_y)
                self.invalidate_delay = TOTAL_INVALIDATE_DELAY
                self.scroll_delay = 3
            elif scroll_y > 0:
                self.proj.zoomout(self.mouse_x, self.mouse_y)
                self.invalidate_delay = TOTAL_INVALIDATE_DELAY
                self.scroll_delay = 3


    def on_key_release(self, symbol, modifiers):
        if symbol == pyglet.window.key.P:
            fname = '%d.png' % (time.time()*1000)
            GeoplotlibApp.screenshot(fname)
            print(fname + ' saved')
        elif symbol == pyglet.window.key.M:
            self.show_map = not self.show_map
        elif symbol == pyglet.window.key.L:
            self.show_layers = not self.show_layers
        elif symbol == pyglet.window.key.I:
            self.proj.zoomin(SCREEN_W/2, SCREEN_H/2)
            self.invalidate_delay = TOTAL_INVALIDATE_DELAY
        elif symbol == pyglet.window.key.O:
            self.proj.zoomout(SCREEN_W/2, SCREEN_H/2)
            self.invalidate_delay = TOTAL_INVALIDATE_DELAY
        elif symbol == pyglet.window.key.R:
            # hack to force invalidate
            self.invalidate_delay = 3
        elif symbol == pyglet.window.key.A:
            self.proj.pan(-KEYBOARD_PAN, 0)
        elif symbol == pyglet.window.key.D:
            self.proj.pan(+KEYBOARD_PAN, 0)
        elif symbol == pyglet.window.key.W:
            self.proj.pan(0, +KEYBOARD_PAN)
        elif symbol == pyglet.window.key.S:
            self.proj.pan(0, -KEYBOARD_PAN)
        elif symbol == pyglet.window.key.B:
            print(self.proj.bbox())
        elif symbol == pyglet.window.key.C:
            self.show_coordinates = not self.show_coordinates
        else:
            for l in self.geoplotlib_config.layers:
                need_invalidate = l.on_key_release(symbol, modifiers)
                if need_invalidate:
                    l.invalidate(self.proj)


    @staticmethod
    def screenshot(fname):
        glPixelTransferf(gl.GL_ALPHA_BIAS, 1.0)
        image = pyglet.image.ColorBufferImage(0, 0, SCREEN_W, SCREEN_H)
        image.save(fname)
        glPixelTransferf(gl.GL_ALPHA_BIAS, 0.0)


    def on_update(self, dt):
        self.ticks += dt*1000


    def start(self):
        #pyglet.options['debug_gl'] = False
        if self.geoplotlib_config.bbox is not None:
            self.proj.fit(self.geoplotlib_config.bbox, force_zoom=self.geoplotlib_config.requested_zoom)
        elif len(self.geoplotlib_config.layers) > 0:
            self.proj.fit(BoundingBox.from_bboxes([l.bbox() for l in self.geoplotlib_config.layers]),
                          force_zoom=self.geoplotlib_config.requested_zoom)
        for l in self.geoplotlib_config.layers:
            l.invalidate(self.proj)

        pyglet.app.run()


def _flatten_xy(x, y):
        return np.vstack((x, y)).T.flatten()


class BatchPainter:
    """
    This class batches OpenGL calls. The usage pattern is to instantiate a BatchPainter,
     perform all the drawing and finally render using batch_draw
    """

    def __init__(self):
        self._batch = pyglet.graphics.Batch()
        self._color = [0, 0, 255, 255]
        self._sprites = []
        self._labels = []


    def set_color(self, color):
        if color == 'k' or color == 'black':
            self._color = [0,0,0,255]
        elif color == 'w' or color == 'white':
            self._color = [255,255,255,255]
        elif color == 'r' or color == 'red':
            self._color = [255,0,0,255]
        elif color == 'g' or color == 'green':
            self._color = [0,255,0,255]
        elif color == 'b' or color == 'blue':
            self._color = [0,0,255,255]
        elif len(color) == 4:
            for c in color:
                if c < 0 or c > 255:
                    raise Exception('color components must be between 0 and 255')
            self._color = color
        elif len(color) == 3:
            for c in color:
                if c < 0 or c > 255:
                    raise Exception('color components must be between 0 and 255')
            self._color = color + [255]
        else:
            raise Exception('invalid color format')


    def lines(self, x0, y0, x1, y1, colors=None, width=1.0):
        glLineWidth(width)
        x = _flatten_xy(x0, x1)
        y = _flatten_xy(y0, y1)
        vertices = _flatten_xy(x, y)
        if colors is None:
            colors = self._color * int(len(vertices)/VERT_PER_POINT)

        self._batch.add(int(len(vertices)/VERT_PER_POINT), GL_LINES, None,
                      ('v2f', vertices),
                      ('c4B', np.array(colors).flatten()))


    def linestrip(self, x, y, width=1.0, closed=False):
        glLineWidth(width)
        vertices = _flatten_xy(x, y)
        indices = [i // 2 for i in range(len(vertices))]
        indices = indices[1:-1]
        if closed:
            indices.append(indices[-1])
            indices.append(indices[0])

        self._batch.add_indexed(int(len(vertices)/VERT_PER_POINT), GL_LINES, None,
                      indices,
                      ('v2f', vertices),
                      ('c4B', self._color * int(len(vertices)/VERT_PER_POINT)))


    def poly(self, x, y, width=1.0):
        glLineWidth(width)
        vertices = _flatten_xy(x, y)
        indices = []
        for i in range(1, len(x) - 1):
            indices.append(0)
            indices.append(i)
            indices.append(i+1)

        self._batch.add_indexed(int(len(vertices)/VERT_PER_POINT), GL_TRIANGLES, None,
                      indices,
                      ('v2f', vertices),
                      ('c4B', self._color * int(len(vertices)/VERT_PER_POINT)))


    def triangle(self, vertices):
        self._batch.add(int(len(vertices)/VERT_PER_POINT), GL_TRIANGLES, None,
                      ('v2f', vertices),
                      ('c4B', self._color * int(len(vertices)/VERT_PER_POINT)))


    def circle(self, cx, cy, r, width=2.0):
        glLineWidth(width)

        precision = int(10*math.log(r))

        vertices = []
        for alpha in np.linspace(0, 6.28, precision):
            vertices.append(cx + r * math.cos(alpha))
            vertices.append(cy + r * math.sin(alpha))

        indices = []
        for i in range(precision - 1):
            indices.append(i)
            indices.append(i+1)
        indices.append(precision-1)
        indices.append(0)

        self._batch.add_indexed(int(len(vertices)/VERT_PER_POINT), GL_LINES, None,
                      indices,
                      ('v2f', vertices),
                      ('c4B', self._color * int(len(vertices)/VERT_PER_POINT)))


    def circle_filled(self, cx, cy, r):
        vertices = []
        vertices.append(cx)
        vertices.append(cy)

        precision = int(10*math.log(r))

        for alpha in np.linspace(0, 6.28, precision):
            vertices.append(cx + r * math.cos(alpha))
            vertices.append(cy + r * math.sin(alpha))

        indices = []
        for i in range(1, precision):
            indices.append(0)
            indices.append(i)
            indices.append(i+1)
        indices.append(0)
        indices.append(precision)
        indices.append(1)

        self._batch.add_indexed(int(len(vertices)/VERT_PER_POINT), GL_TRIANGLES, None,
                      indices,
                      ('v2f', vertices),
                      ('c4B', self._color * int(len(vertices)/VERT_PER_POINT)))


    def points(self, x, y, point_size=10, rounded=False):
        glPointSize(point_size)
        if rounded:
            glEnable(GL_POINT_SMOOTH)
        else:
            glDisable(GL_POINT_SMOOTH)

        vertices = np.vstack((x, y)).T.flatten()

        self._batch.add(int(len(vertices)/VERT_PER_POINT), GL_POINTS, None,
                        ('v2f', vertices),
                        ('c4B', self._color * int(len(vertices)/VERT_PER_POINT)))


    def rect(self, left, top, right, bottom):
        self.triangle([left, top, right, top, right, bottom, right, bottom, left, top, left, bottom])


    def batch_rects(self, rects_vertices, rects_colors):
        triangles = []
        colors = []
        for i in range(len(rects_vertices)):
            r = rects_vertices[i]
            c = rects_colors[i]
            left, top, right, bottom = r
            triangles.extend([left, top, right, top, right, bottom, right, bottom, left, top, left, bottom])
            colors.extend(c * 6)

        self._batch.add(int(len(triangles)/VERT_PER_POINT), GL_TRIANGLES, None,
                      ('v2f', triangles),
                      ('c4B', colors))


    def sprites(self, image, x, y, scale=1.0):
        from pyglet.sprite import Sprite
        for i in range(len(x)):
            sprite = Sprite(image, batch=self._batch)
            sprite.x = x[i]
            sprite.y = y[i]
            sprite.scale = scale
            self._sprites.append(sprite)


    def labels(self, x, y, texts, font_name=FONT_NAME, font_size=14, anchor_x='left', anchor_y='top'):
        for i in range(len(x)):
            lab = pyglet.text.Label(parse_raw_str(texts if type(texts) == str else texts[i]),
                                    batch=self._batch,
                                    color=self._color,
                                    font_name=font_name,
                                    font_size=font_size,
                                    x=x[i], y=y[i],
                                    anchor_x=anchor_x, 
                                    anchor_y=anchor_y)
            self._labels.append(lab)


    def convexhull(self, x, y, fill=False, smooth=False):
        try:
            from scipy.spatial import ConvexHull
            from scipy.spatial.qhull import QhullError
        except:
            raise Exception('ConvexHull requires scipy')

        if len(x) < 3:
            raise Exception('convexhull requires at least 3 points')


        points = np.vstack((x,y)).T
        try:
            hull = ConvexHull(points)
            xhull = points[hull.vertices,0]
            yhull = points[hull.vertices,1]

            if smooth:
                xhull, yhull = self.__generate_spline(xhull, yhull, closed=True)

            if fill:
                self.poly(xhull,yhull)
            else:
                self.linestrip(xhull, yhull, 3, closed=True)

        except QhullError as qerr:
            self.linestrip(x, y, 3, closed=False)


    def __generate_spline(self, x, y, closed=False, steps=20):
        """
        catmullrom spline
        http://www.mvps.org/directx/articles/catmull/
        """

        if closed:
            x = x.tolist()
            x.insert(0, x[-1])
            x.append(x[1])
            x.append(x[2])

            y = y.tolist()
            y.insert(0, y[-1])
            y.append(y[1])
            y.append(y[2])

        points = np.vstack((x,y)).T

        curve = []

        if not closed:
            curve.append(points[0])

        for j in range(1, len(points)-2):
            for s in range(steps):
                t = 1. * s / steps
                p0, p1, p2, p3 = points[j-1], points[j], points[j+1], points[j+2]
                pnew = 0.5 *((2 * p1) + (-p0 + p2) * t + (2*p0 - 5*p1 + 4*p2 - p3) * t**2 + (-p0 + 3*p1- 3*p2 + p3) * t**3)
                curve.append(pnew)

        if not closed:
            curve.append(points[-1])

        curve = np.array(curve)
        return curve[:, 0], curve[:, 1]


    def spline(self, x, y, width=3):
        xcurve, ycurve = self.__generate_spline(x, y, closed=False)
        self.linestrip(xcurve, ycurve, width)


    def batch_draw(self):
        self._batch.draw()


class Projector():

    def __init__(self):
        self.tiles_horizontally = 1.*SCREEN_W / TILE_SIZE
        self.tiles_vertically = 1.*SCREEN_H / TILE_SIZE
        self.fit(BoundingBox.WORLD)


    def set_to(self, north, west, zoom):
        self.zoom = zoom
        self.xtile, self.ytile = self.deg2num(north, west, zoom)


    def fit(self, bbox, max_zoom=MAX_ZOOM, force_zoom=None):
        """
        Fits the projector to a BoundingBox
        :param bbox: BoundingBox
        :param max_zoom: max zoom allowed
        :param force_zoom: force this specific zoom value even if the whole bbox does not completely fit
        """

        BUFFER_FACTOR = 1.1
        
        if force_zoom is not None:
            self.zoom = force_zoom
        else:
            for zoom in range(max_zoom, MIN_ZOOM-1, -1):
                self.zoom = zoom
                left, top = self.lonlat_to_screen([bbox.west], [bbox.north])
                right, bottom = self.lonlat_to_screen([bbox.east], [bbox.south])
                if (top - bottom < SCREEN_H*BUFFER_FACTOR) and (right - left < SCREEN_W*BUFFER_FACTOR):
                    break

        west_tile, north_tile = self.deg2num(bbox.north, bbox.west, self.zoom)
        east_tile, south_tile = self.deg2num(bbox.south, bbox.east, self.zoom)
        self.xtile = west_tile - self.tiles_horizontally/2. + (east_tile - west_tile)/2
        self.ytile = north_tile - self.tiles_vertically/2. + (south_tile - north_tile)/2
        self.calculate_viewport_size()


    @staticmethod
    def deg2num(lat_deg, lon_deg, zoom):
        lat_rad = math.radians(lat_deg)
        n = 2.0 ** zoom
        xtile = (lon_deg + 180.0) / 360.0 * n
        ytile = (1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n
        return (xtile, ytile)


    @staticmethod
    def num2deg(xtile, ytile, zoom):
        n = 2.0 ** zoom
        lon_deg = xtile / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
        lat_deg = math.degrees(lat_rad)
        return (lat_deg, lon_deg)


    def bbox(self):
        north, west = self.num2deg(self.xtile, self.ytile, self.zoom)
        south, east = self.num2deg(self.xtile + self.tiles_horizontally, self.ytile + self.tiles_vertically, self.zoom)
        return BoundingBox(north=north, west=west, south=south, east=east)


    def pan(self, deltax, deltay):
        self.xtile += deltax
        self.ytile -= deltay


    def zoomin(self, mouse_x, mouse_y):
        mouse_lat, mouse_lon = self.screen_to_latlon(mouse_x, mouse_y)
        self.zoom = min(self.zoom + 1, MAX_ZOOM)
        self.xtile, self.ytile = self.deg2num(mouse_lat, mouse_lon, self.zoom)
        self.xtile -= 1. * mouse_x / TILE_SIZE
        self.ytile -= 1. * mouse_y / TILE_SIZE
        self.calculate_viewport_size()


    def zoomout(self, mouse_x, mouse_y):
        mouse_lat, mouse_lon = self.screen_to_latlon(mouse_x, mouse_y)
        self.zoom = max(self.zoom - 1, MIN_ZOOM)
        self.xtile, self.ytile = self.deg2num(mouse_lat, mouse_lon, self.zoom)
        self.xtile -= 1. * mouse_x / TILE_SIZE
        self.ytile -= 1. * mouse_y / TILE_SIZE
        self.calculate_viewport_size()


    def calculate_viewport_size(self):
        lat1, lon1 = Projector.num2deg(self.xtile, self.ytile, self.zoom)
        lat2, lon2 = Projector.num2deg(self.xtile + self.tiles_horizontally, self.ytile + self.tiles_vertically, self.zoom)
        self.viewport_w = utils.haversine(lat1=lat1, lon1=lon1, lat2=lat1, lon2=lon2)
        self.viewport_h = utils.haversine(lat1=lat1, lon1=lon1, lat2=lat2, lon2=lon1)


    def lonlat_to_screen(self, lon, lat):
        """
        Projects geodesic coordinates to screen
        :param lon: longitude
        :param lat: latitude
        :return: x,y screen coordinates
        """
        if type(lon) == list:
            lon = np.array(lon)
        if type(lat) == list:
            lat = np.array(lat)

        lat_rad = np.radians(lat)
        n = 2.0 ** self.zoom
        xtile = (lon + 180.0) / 360.0 * n
        ytile = (1.0 - np.log(np.tan(lat_rad) + (1 / np.cos(lat_rad))) / math.pi) / 2.0 * n
        x = (xtile * TILE_SIZE).astype(int)
        y = (SCREEN_H - ytile * TILE_SIZE).astype(int)
        return x, y


    def screen_to_latlon(self, x, y):
        """
        Return the latitude and longitude corresponding to a screen point
        :param x: screen x
        :param y: screen y
        :return: latitude and longitude at x,y
        """
        xtile = 1. * x / TILE_SIZE + self.xtile
        ytile = 1. * y / TILE_SIZE + self.ytile
        return self.num2deg(xtile, ytile, self.zoom)


class SetQueue(Queue):

    def _init(self, maxsize):
        self.queue = set()

    def _put(self, item):
        self.queue.add(item)

    def _get(self):
        return self.queue.pop()


class TileDownloaderThread(Thread):

    def __init__(self, queue):
        Thread.__init__(self)
        self.queue = queue
        self.daemon = True

    def run(self):
        while True:
            url, download_path = self.queue.get()
            assert download_path.endswith('.png')
            try:
                # print "downloading %s as %s" % (url, download_path)
                source = urllib2.urlopen(url)
                content = source.read()
                source.close()
                destination = open(download_path,'wb')
                destination.write(content)
                destination.close()
            except Exception as e:
                print(url, e)


_GEOPLOTLIB_ATTRIBUTION = u'made with geoplotlib | '

_DEFAULT_TILE_PROVIDERS = {
    'watercolor': { 'url': lambda zoom, xtile, ytile:
                            'http://%s.tile.stamen.com/watercolor/%d/%d/%d.png' % (random.choice(['a', 'b', 'c', 'd']), zoom, xtile, ytile),
                    'attribution': _GEOPLOTLIB_ATTRIBUTION + 'Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap, under ODbL.'
    },
    'toner': { 'url': lambda zoom, xtile, ytile:
                        "http://%s.tile.stamen.com/toner/%d/%d/%d.png" % (random.choice(['a', 'b', 'c', 'd']), zoom, xtile, ytile),
               'attribution': _GEOPLOTLIB_ATTRIBUTION + 'Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap, under ODbL.'
    },
    'toner-lite': { 'url': lambda zoom, xtile, ytile:
                            "http://%s.tile.stamen.com/toner-lite/%d/%d/%d.png" % (random.choice(['a', 'b', 'c', 'd']), zoom, xtile, ytile),
                    'attribution': _GEOPLOTLIB_ATTRIBUTION + 'Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap, under ODbL.'
    },
    'darkmatter': { 'url': lambda zoom, xtile, ytile:
                            'http://%s.basemaps.cartocdn.com/dark_all/%d/%d/%d.png' % (random.choice(['a', 'b', 'c']), zoom, xtile, ytile),
                    'attribution': _GEOPLOTLIB_ATTRIBUTION + u'© OpenStreetMap contributors © CartoDB'
    },
    'positron': { 'url': lambda zoom, xtile, ytile:
                            'http://%s.basemaps.cartocdn.com/light_all/%d/%d/%d.png' % (random.choice(['a', 'b', 'c']), zoom, xtile, ytile),
                    'attribution': _GEOPLOTLIB_ATTRIBUTION + u'© OpenStreetMap contributors © CartoDB'
    }

}

class MapLayer():

    def __init__(self, tiles_provider, skipdl=False):
        if type(tiles_provider) == str:
            if tiles_provider in _DEFAULT_TILE_PROVIDERS:
                self.tiles_dir = tiles_provider
                self.url_generator = _DEFAULT_TILE_PROVIDERS[tiles_provider]['url']
                self.attribution = _DEFAULT_TILE_PROVIDERS[tiles_provider]['attribution']
            else:
                raise Exception('unknown style ' + tiles_provider)
        else:
            self.tiles_dir = tiles_provider['tiles_dir']
            self.url_generator = tiles_provider['url']
            self.attribution = tiles_provider['attribution']

        self.skipdl = skipdl
        self.tiles_cache = {}
        self.download_queue = SetQueue()
        self.download_threads = [TileDownloaderThread(self.download_queue) for i in range(2)]
        for t in self.download_threads:
            t.start()


    def get_tile(self, zoom, xtile, ytile):
        if xtile < 0 or ytile < 0 or xtile >= 2**zoom or ytile >= 2**zoom:
            return None

        tile_image = self.tiles_cache.get((zoom, xtile, ytile))
        if tile_image is not None:
            return tile_image

        url = self.url_generator(zoom, xtile, ytile)
        dir_path = expanduser('~') + '/geoplotlib_tiles/%s/%d/%d/' % (self.tiles_dir, zoom, xtile)
        download_path = dir_path + '%d.png' % ytile

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        if not os.path.isfile(download_path):
            if self.skipdl:
                return None
            else:
                self.download_queue.put((url, download_path))
        else:
            try:
                tile_image = pyglet.image.load(download_path)
                tile_image.blit(2*SCREEN_W, 2*SCREEN_H, 0) # blit offscreen to check if valid
                self.tiles_cache[(zoom, xtile, ytile)] = pyglet.sprite.Sprite(tile_image)
                return self.tiles_cache[(zoom, xtile, ytile)]
            except Exception as exc:
                print(exc)
                assert download_path.endswith('.png')
                os.unlink(download_path)
                return None


    def draw(self, proj):
        for x in range(int(proj.xtile), int(proj.xtile + proj.tiles_horizontally + 1)):
            for y in range(int(proj.ytile), int(proj.ytile + proj.tiles_vertically + 1)):
                tilesurf = self.get_tile(proj.zoom, x, y)
                if tilesurf is not None:
                    try:
                        tilesurf.x = int((x - proj.xtile)*TILE_SIZE)
                        tilesurf.y = int(SCREEN_H - (y - proj.ytile + 1)*TILE_SIZE)
                        tilesurf.draw()
                    except Exception as e:
                        print('exception blitting', x, y, proj.zoom, e)
