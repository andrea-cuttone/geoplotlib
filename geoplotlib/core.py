from Queue import Queue
from inspect import isfunction
from threading import Thread
import pyglet
from pyglet.gl import *
from pyglet.sprite import Sprite
from pyglet.window import mouse
import time
from pyglet.gl import *
from collections import namedtuple
import math
import numpy as np
import os
import random
import urllib2
import pyglet
from os.path import expanduser
from geoplotlib.utils import BoundingBox, parse_raw_str


VERT_PER_POINT = 2
SCREEN_W = 1280
SCREEN_H = 768
FPS = 60
TILE_SIZE = 256
MIN_ZOOM = 2
MAX_ZOOM = 20


class UiManager:

    def __init__(self):
        self.font_color = (0,0,0,255)
        self.font_size = 20
        self.font_name = 'Times New Roman'
        self.padding = 2

        self.labels = {}

        self.labels['status'] = pyglet.text.Label('',
                                       color=self.font_color,
                                       font_name=self.font_name,
                                       font_size=self.font_size,
                                       x=20, y=10,
                                       anchor_x='left', anchor_y='bottom')

        self.labels['tooltip'] = pyglet.text.Label('',
                                       color=self.font_color,
                                       font_name=self.font_name,
                                       font_size=self.font_size,
                                       x=SCREEN_W, y=SCREEN_H,
                                       anchor_x='left', anchor_y='bottom')

        self.labels['info'] = pyglet.text.Label('',
                                       color=self.font_color,
                                       font_name=self.font_name,
                                       font_size=self.font_size,
                                       x=SCREEN_W, y=SCREEN_H,
                                       anchor_x='right', anchor_y='top')

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


class BaseApp(pyglet.window.Window):

    def __init__(self):
        super(BaseApp, self).__init__(SCREEN_W, SCREEN_H)
        self.ticks = 0
        self.ui_manager = UiManager()
        self.proj = Projector()
        self.map_layer = MapLayer('toner', skipdl=False)
        self._layers = []

        self.scroll_delay = 0
        self.drag_x = self.drag_y = 0
        self.dragging = False
        self.drag_start_timestamp = 0
        self.autofit = True

        self.mouse_x = self.mouse_y = 0

        glEnable(GL_LINE_SMOOTH);
        glEnable(GL_POLYGON_SMOOTH);
        # glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
        # glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        pyglet.clock.schedule_interval(self.on_update, 1. / FPS)


    def add_layer(self, layer):
        self._layers.append(layer)


    def clear_layers(self):
        self._layers = []


    def on_draw(self):
        self.clear()

        # needed to avoid diagonal artifacts on the tiles
        glDisable(GL_LINE_SMOOTH)
        glDisable(GL_POLYGON_SMOOTH)

        self.ui_manager.clear()

        self.map_layer.draw(self.proj)

        # glEnable(GL_LINE_SMOOTH);
        # glEnable(GL_POLYGON_SMOOTH);

        glPushMatrix()
        glTranslatef(-self.proj.xtile * TILE_SIZE, self.proj.ytile * TILE_SIZE, 0)
        for l in self._layers:
            l.draw(self.mouse_x + self.proj.xtile * TILE_SIZE,
                   SCREEN_H - self.mouse_y - self.proj.ytile * TILE_SIZE,
                   self.ui_manager)
        glPopMatrix()

        self.ui_manager.status('T: %.1f, FPS:%d' % (self.ticks / 1000., pyglet.clock.get_fps()))
        self.ui_manager.draw(self.mouse_x, SCREEN_H - self.mouse_y)


    def on_mouse_motion(self, x, y, dx, dy):
        self.mouse_x = x
        self.mouse_y = SCREEN_H - y


    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if buttons & mouse.LEFT:
            self.drag_start_timestamp = self.ticks
            self.drag_x = -1. * dx / TILE_SIZE
            self.drag_y = -1. * dy / TILE_SIZE
            self.proj.pan(self.drag_x, self.drag_y)


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


    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        if self.scroll_delay == 0:
            if scroll_y < 0:
                self.proj.zoomin(self.mouse_x, self.mouse_y)
                for l in self._layers:
                    l.invalidate(self.proj)
                self.scroll_delay = 60
            elif scroll_y > 0:
                self.proj.zoomout(self.mouse_x, self.mouse_y)
                for l in self._layers:
                    l.invalidate(self.proj)
                self.scroll_delay = 60


    def on_key_release(self, symbol, modifiers):
        if symbol == pyglet.window.key.S:
            fname = '%d.png' % (time.time()*1000)
            pyglet.image.get_buffer_manager().get_color_buffer().save(fname)
            print fname + ' saved'


    def on_update(self, dt):
        self.ticks += dt*1000

        if self.scroll_delay > 0:
            self.scroll_delay -= 1

        if abs(self.drag_x) > 1e-3 or abs(self.drag_y) > 1e-3:
            self.drag_x *= 0.93
            self.drag_y *= 0.93

            if self.dragging == False:
                self.proj.pan(self.drag_x, self.drag_y)

        for l in self._layers:
            if hasattr(l, 'on_tick'):
                l.on_tick(dt, self.proj)


    def run(self):
        #pyglet.options['debug_gl'] = False
        if self.autofit:
            self.proj.fit(BoundingBox.from_bboxes([l.bbox() for l in self._layers]))

        for l in self._layers:
            l.invalidate(self.proj)

        pyglet.app.run()


    def set_bbox(self, bbox):
        self.proj.fit(bbox)
        self.autofit = False


def _flatten_xy(x, y):
        return np.vstack((x, y)).T.flatten()


class BatchPainter:

    def __init__(self):
        self._batch = pyglet.graphics.Batch()
        self._color = [0, 0, 0, 255]
        self._sprites = []


    def set_color(self, color):
        if len(color) == 4:
            self._color = color
        elif len(color) == 3:
            self._color = color + [255]
        else:
            raise Exception('invalid color format')


    def lines(self, x0, y0, x1, y1, width=1.0):
        glLineWidth(width)
        x = _flatten_xy(x0, x1)
        y = _flatten_xy(y0, y1)
        vertices = _flatten_xy(x, y)
        self._batch.add(len(vertices)/VERT_PER_POINT, GL_LINES, None,
                      ('v2f', vertices),
                      ('c4B', self._color * (len(vertices)/VERT_PER_POINT)))


    def linestrip(self, x, y, width=1.0, closed=False):
        glLineWidth(width)
        vertices = _flatten_xy(x, y)
        indices = [i // 2 for i in range(len(vertices))]
        indices = indices[1:-1]
        if closed:
            indices.append(indices[-1])
            indices.append(indices[0])

        self._batch.add_indexed(len(vertices)/VERT_PER_POINT, GL_LINES, None,
                      indices,
                      ('v2f', vertices),
                      ('c4B', self._color * (len(vertices)/VERT_PER_POINT)))


    def triangle(self, vertices):
        self._batch.add(len(vertices)/VERT_PER_POINT, GL_TRIANGLES, None,
                      ('v2f', vertices),
                      ('c4B', self._color * (len(vertices)/VERT_PER_POINT)))


    def points(self, x, y, point_size=10, rounded=False):
        glPointSize(point_size)
        if rounded:
            glEnable(GL_POINT_SMOOTH)
        else:
            glDisable(GL_POINT_SMOOTH)

        # TODO: ravel? http://stackoverflow.com/questions/9057379/correct-and-efficient-way-to-flatten-array-in-numpy-in-python
        vertices = np.vstack((x, y)).T.flatten()

        self._batch.add(len(vertices)/VERT_PER_POINT, GL_POINTS, None,
                      ('v2f', vertices),
                      ('c4B', self._color * (len(vertices)/VERT_PER_POINT)))


    def rect(self, left, top, right, bottom):
        self.triangle([left, top, right, top, right, bottom, right, bottom, left, top, left, bottom])


    def sprites(self, image, x, y, scale=1.0):
        for i in range(len(x)):
            sprite = Sprite(image, batch=self._batch)
            sprite.x = x[i]
            sprite.y = y[i]
            sprite.scale = scale
            self._sprites.append(sprite)


    def batch_draw(self):
        self._batch.draw()


class Projector():

    def __init__(self):
        self.tiles_horizontally = int(math.ceil(1.*SCREEN_W / TILE_SIZE))
        self.tiles_vertically = int(math.ceil(1.*SCREEN_H / TILE_SIZE))
        self.fit(BoundingBox.WORLD)


    def set_to(self, north, west, zoom):
        self.zoom = zoom
        self.xtile, self.ytile = self.deg2num(north, west, zoom)


    def fit(self, bbox):
        for zoom in range(MAX_ZOOM, MIN_ZOOM-1, -1):
            self.zoom = zoom
            left, top = self.lonlat_to_screen([bbox.west], [bbox.north])
            right, bottom = self.lonlat_to_screen([bbox.east], [bbox.south])
            if (top - bottom < SCREEN_H) and (right - left < SCREEN_W):
                break
        #self.zoom = zoom - 1
        west_tile, north_tile = self.deg2num(bbox.north, bbox.west, self.zoom)
        east_tile, south_tile = self.deg2num(bbox.south, bbox.east, self.zoom)
        self.xtile = west_tile - self.tiles_horizontally/2. + (east_tile - west_tile)/2
        self.ytile = north_tile - self.tiles_vertically/2. + (south_tile - north_tile)/2


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
        #print 'moving to', self.nw, self.zoom


    def pan(self, deltax, deltay):
        self.xtile += deltax
        self.ytile -= deltay


    def zoomin(self, mouse_x, mouse_y):
        mouse_lat, mouse_lon = self.screen_to_latlon(mouse_x, mouse_y)
        self.zoom = min(self.zoom + 1, MAX_ZOOM)
        self.xtile, self.ytile = self.deg2num(mouse_lat, mouse_lon, self.zoom)
        self.xtile -= 1. * mouse_x / TILE_SIZE
        self.ytile -= 1. * mouse_y / TILE_SIZE


    def zoomout(self, mouse_x, mouse_y):
        mouse_lat, mouse_lon = self.screen_to_latlon(mouse_x, mouse_y)
        self.zoom = max(self.zoom - 1, MIN_ZOOM)
        self.xtile, self.ytile = self.deg2num(mouse_lat, mouse_lon, self.zoom)
        self.xtile -= 1. * mouse_x / TILE_SIZE
        self.ytile -= 1. * mouse_y / TILE_SIZE


    def lonlat_to_screen(self, lon, lat):
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
            try:
                print "downloading %s as %s" % (url, download_path)
                source = urllib2.urlopen(url)
                content = source.read()
                source.close()
                destination = open(download_path,'wb')
                destination.write(content)
                destination.close()
            except Exception as e:
                print e


class MapLayer():

    def __init__(self, tiles_provider, skipdl=False):
        self.tiles_provider = tiles_provider
        self.skipdl = skipdl
        self.tiles_cache = {}
        self.download_queue = SetQueue()
        self.download_threads = [TileDownloaderThread(self.download_queue).start() for i in range(4)]


    def get_tile(self, zoom, xtile, ytile):
        tile_image = self.tiles_cache.get((zoom, xtile, ytile))
        if tile_image is not None:
            return tile_image

        tiles_dir = self.tiles_provider
        if self.tiles_provider == 'watercolor':
            url = 'http://%s.tile.stamen.com/watercolor/%d/%d/%d.png' % (random.choice(['a', 'b', 'c', 'd']), zoom, xtile, ytile)
        elif self.tiles_provider == 'toner':
            url = "http://%s.tile.stamen.com/toner/%d/%d/%d.png" % (random.choice(['a', 'b', 'c', 'd']), zoom, xtile, ytile)
        elif self.tiles_provider == 'toner-lite':
            url = "http://%s.tile.stamen.com/toner/%d/%d/%d.png" % (random.choice(['a', 'b', 'c', 'd']), zoom, xtile, ytile)
        elif self.tiles_provider == 'mapquest':
            url = "http://otile%d.mqcdn.com/tiles/1.0.0/osm/%d/%d/%d.png" % (random.randint(1, 4), zoom, xtile, ytile)
        elif self.tiles_provider == 'toolserver':
            url = 'http://%s.www.toolserver.org/tiles/bw-mapnik/%d/%d/%d.png' % (random.choice(['a', 'b', 'c']), zoom, xtile, ytile)
        elif isfunction(self.tiles_provider):
            url = self.tiles_provider(zoom, xtile, ytile)
            tiles_dir = 'custom'
        else:
            raise Exception('unknown style')

        dir_path = expanduser('~') + '/geoplotlib_tiles/%s/%d/%d/' % (tiles_dir, zoom, xtile)
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
                self.tiles_cache[(zoom, xtile, ytile)] = tile_image
                return tile_image
            except Exception as e:
                print 'exception downloading', download_path, e
                os.unlink(download_path)
                return None


    def draw(self, proj):
        for x in range(int(proj.xtile), int(proj.xtile) + proj.tiles_horizontally + 1):
            for y in range(int(proj.ytile), int(proj.ytile) + proj.tiles_vertically + 1):
                tilesurf = self.get_tile(proj.zoom, x, y)
                if tilesurf is not None:
                    x_screen = int((x - proj.xtile)*TILE_SIZE)
                    y_screen = int(SCREEN_H - (y - proj.ytile + 1)*TILE_SIZE)
                    try:
                        tilesurf.blit(x_screen, y_screen, 0)
                    except Exception as e:
                        print 'exception blitting', x, y, proj.zoom, e
