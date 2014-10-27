import pyglet
from geoplotlib.layers import BaseLayer
import pandas as pd
from geoplotlib.core import BatchPainter, MAX_ZOOM
import geoplotlib
from geoplotlib.utils import epoch_to_str, BoundingBox


class CustomLayer(BaseLayer):

    def __init__(self, data):
        self.data = data
        self.idx = 1
        self.painter = BatchPainter()


    def invalidate(self, proj):
        self.painter = BatchPainter()
        proj.fit(BoundingBox.from_points(lons=self.data.iloc[max(self.idx-5,0):self.idx]['lon'],
                                         lats=self.data.iloc[max(self.idx-5,0):self.idx]['lat']),
                 max_zoom=18)
        df = self.data.iloc[:self.idx]
        x, y = proj.lonlat_to_screen(df.lon, df.lat)
        self.painter.set_color([255,0,0])
        self.painter.lines(x[:-1], y[:-1], x[1:], y[1:])
        self.painter.points(x, y, 7)
        self.painter.set_color([0,0,255])
        self.painter.points(x[-1], y[-1], 10)


    def draw(self, mouse_x, mouse_y, ui_manager):
        self.painter.batch_draw()
        ui_manager.info(epoch_to_str(self.data.iloc[self.idx].timestamp))


    def bbox(self):
        return BoundingBox.KBH


    def on_key_release(self, key, modifiers):
        if key == pyglet.window.key.LEFT:
            self.idx = self.idx - 1
            return True
        elif key == pyglet.window.key.RIGHT:
            self.idx = self.idx + 1
            return True
        return False


geoplotlib.add_layer(CustomLayer(pd.read_csv('somedata.csv')))
geoplotlib.show()
