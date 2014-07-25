import sys,os
sys.path.append(os.path.realpath('..'))

import pandas as pd
from geoplotlib.core import BatchPainter
import geoplotlib
from geoplotlib.colors import colorbrewer
from geoplotlib.utils import epoch_to_str


class TrailsLayer():

    def __init__(self):
        self.data = pd.read_csv('data/taxi.csv') # TODO: remove pandas dependency
        self.data = self.data[(self.data.lon > 115.950) &
                              (self.data.lon < 116.796) &
                              (self.data.lat < 40.212) &
                              (self.data.lat > 39.631)
        ]
        self.cmap = colorbrewer(self.data.taxi_id, alpha=220)
        self.t = self.data.timestamp.min()


    def invalidate(self, proj):
        pass


    def draw(self, mouse_x, mouse_y, ui_manager):
        self.painter.batch_draw()
        ui_manager.info(epoch_to_str(self.t))


    def on_tick(self, dt, proj):
        self.painter = BatchPainter()
        df = self.data[(self.data.timestamp > self.t) & (self.data.timestamp <= self.t + 15*60)]
        proj.fit(df.lon.values, df.lat.values)
        for taxi_id, grp in df.groupby('taxi_id'):
            self.painter.set_color(self.cmap[taxi_id])
            x, y = proj.lonlat_to_screen(grp.lon.values, grp.lat.values)
            self.painter.points(x, y, 10)

        self.t += 5*60*dt


geoplotlib.add_layer(TrailsLayer())
geoplotlib.tiles_provider(lambda zoom, xtile, ytile: "http://a.tile.stamen.com/toner/%d/%d/%d.png" % (zoom, xtile, ytile))
geoplotlib.show()
