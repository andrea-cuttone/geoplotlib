import sys,os
from pyglet.gl import glPushMatrix, glPopMatrix
from geoplotlib.layers import HotspotManager

sys.path.append(os.path.realpath('..'))

from geoplotlib.core import BatchPainter
import geoplotlib
import numpy as np

import shapefile


class Shape():

    def __init__(self, points, attr):
        self.points = np.array(points)
        self.attr = attr


    def lon(self):
        return self.points[:,0]


    def lat(self):
        return self.points[:,1]


class PolyLayer():

    def __init__(self):
        reader = shapefile.Reader('data/dk/dk')
        records = reader.shapeRecords()
        self.shapes = []
        for r in records:
            self.shapes.append(Shape(r.shape.points,
                                     {t[0][0]:t[1].decode('latin1') for t in zip(reader.fields[1:], r.record)}))


    def invalidate(self, proj):
        self.painter = BatchPainter()
        self.hotspots = HotspotManager()
        self.painter.set_color([255,0,0])
        for s in self.shapes:
            x, y = proj.lonlat_to_screen(s.lon(), s.lat())
            self.painter.linestrip(x, y, 3, closed=True)
            self.hotspots.add_poly(x, y, s.attr['STEDNAVN'])


    def draw(self, mouse_x, mouse_y, ui_manager):
        self.painter.batch_draw()
        picked = self.hotspots.pick(mouse_x, mouse_y)
        if picked:
            ui_manager.tooltip(picked)


geoplotlib.add_layer(PolyLayer())
geoplotlib.show()
