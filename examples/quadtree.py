"""
Example of a custom layer rendering a quadtree
"""
import geoplotlib
from geoplotlib.layers import BaseLayer
from geoplotlib.core import BatchPainter
from geoplotlib.utils import BoundingBox
import numpy as np


class QuadTree:

    def __init__(self, left, right, top, bottom):
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom


    def split(self):
        middle_x = (self.left + self.right) / 2.
        middle_y = (self.top + self.bottom) / 2.
        return [QuadTree(self.left, middle_x, self.top, middle_y),
                QuadTree(middle_x, self.right, self.top, middle_y),
                QuadTree(self.left, middle_x, middle_y, self.bottom),
                QuadTree(middle_x, self.right, middle_y, self.bottom)]


    def can_split(self, x, y):
        if self.right - self.left < 4:
            return False

        mask = (x > self.left) & (x < self.right) & (y > self.bottom) & (y < self.top)
        return mask.any()


    def __repr__(self):
        return '(%.2f,%.2f,%.2f,%.2f)' % (self.left, self.right, self.top, self.bottom)


class QuadsLayer(BaseLayer):

    def __init__(self, data, cmap='hot_r'):
        self.data = data
        if cmap is not None:
            self.cmap = geoplotlib.colors.ColorMap(cmap, alpha=196)
        else:
            self.cmap = None
            

    def invalidate(self, proj):
        self.painter = BatchPainter()
        x, y = proj.lonlat_to_screen(self.data['lon'], self.data['lat'])
        
        w = x.max() - x.min()
        h = y.max() - y.min()
        w = np.ceil(w / 2) * 2
        h = np.ceil(h / 2) * 2
        l = max(w, h)
        
        root = QuadTree(x.min(), x.min() + l, y.min() + l, y.min())
        maxarea = (root.right - root.left) * (root.top - root.bottom)
        queue = [root]
        done = []
        while len(queue) > 0:
            qt = queue.pop()
            if qt.can_split(x, y):
                queue.extend(qt.split())
            else:
                done.append(qt)
        
        print len(queue), len(done)

        if self.cmap is not None:
            for qt in done:
                area = (qt.right - qt.left) * (qt.top - qt.bottom)
                self.painter.set_color(self.cmap.to_color(1 + area, 1 + maxarea, 'log'))
                self.painter.rect(qt.left, qt.top, qt.right, qt.bottom)
        else:
            for qt in done:
                self.painter.linestrip([qt.left, qt.right, qt.right, qt.left],
                                       [qt.top, qt.top, qt.bottom, qt.bottom], closed=True)
    
            
    def draw(self, proj, mouse_x, mouse_y, ui_manager):
        self.painter.batch_draw()
  

data = geoplotlib.utils.read_csv('data/bus.csv')
geoplotlib.add_layer(QuadsLayer(data, cmap=None))
geoplotlib.set_smoothing(False)
geoplotlib.set_bbox(geoplotlib.utils.BoundingBox.DK)
geoplotlib.show()
