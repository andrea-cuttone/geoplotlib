import sys,os
from geoplotlib.layers import BaseLayer

sys.path.append(os.path.realpath('..'))

import pandas as pd
from geoplotlib.core import BatchPainter
import geoplotlib
from geoplotlib.utils import epoch_to_str, BoundingBox


class CustomLayer(BaseLayer):

    def __init__(self):
        self.data = pd.read_csv('somedata.csv')
        self.t = self.data.timestamp.min()


    def invalidate(self, proj):
        pass


    def draw(self, mouse_x, mouse_y, ui_manager):
        self.painter.batch_draw()
        ui_manager.info(epoch_to_str(self.t))


    def on_tick(self, dt, proj):
        self.painter = BatchPainter()
        self.painter.set_color([255,0,0, 64])
        df = self.data[(self.data.timestamp > self.t) & (self.data.timestamp <= self.t + 60*60*4)]
        for user, grp in df.groupby('user'):
            x, y = proj.lonlat_to_screen(grp.lon.values, grp.lat.values)
            self.painter.linestrip(x, y, 3)
        self.t += 1*60*60*dt


    def bbox(self):
        return BoundingBox.DK


geoplotlib.add_layer(CustomLayer())
geoplotlib.show()
