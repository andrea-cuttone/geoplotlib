"""
Example of keyboard interaction
"""
from geoplotlib.colors import create_set_cmap
import pyglet
from sklearn.cluster import KMeans
import geoplotlib
from geoplotlib.layers import BaseLayer
from geoplotlib.core import BatchPainter
from geoplotlib.utils import BoundingBox
import numpy as np


class KMeansLayer(BaseLayer):

    def __init__(self, data):
        self.data = data
        self.k = 2


    def invalidate(self, proj):
        self.painter = BatchPainter()
        x, y = proj.lonlat_to_screen(self.data['lon'], self.data['lat'])

        k_means = KMeans(n_clusters=self.k)
        k_means.fit(np.vstack([x,y]).T)
        labels = k_means.labels_

        self.cmap = create_set_cmap(set(labels), 'hsv')
        for l in set(labels):
            self.painter.set_color(self.cmap[l])
            self.painter.convexhull(x[labels == l], y[labels == l])
            self.painter.points(x[labels == l], y[labels == l], 2)
    
            
    def draw(self, proj, mouse_x, mouse_y, ui_manager):
        ui_manager.info('Use left and right to increase/decrease the number of clusters. k = %d' % self.k)
        self.painter.batch_draw()


    def on_key_release(self, key, modifiers):
        if key == pyglet.window.key.LEFT:
            self.k = max(2,self.k - 1)
            return True
        elif key == pyglet.window.key.RIGHT:
            self.k = self.k + 1
            return True
        return False
  

data = geoplotlib.utils.read_csv('data/bus.csv')
geoplotlib.add_layer(KMeansLayer(data))
geoplotlib.set_smoothing(True)
geoplotlib.set_bbox(geoplotlib.utils.BoundingBox.DK)
geoplotlib.show()
