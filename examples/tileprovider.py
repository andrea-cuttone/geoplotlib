"""
Example of setting a custom tile provider
"""
import geoplotlib


geoplotlib.tiles_provider({
    'url': lambda zoom, xtile, ytile: 'http://a.tile.stamen.com/watercolor/%d/%d/%d.png' % (zoom, xtile, ytile),
    'tiles_dir': 'mytiles',
    'attribution': 'my attribution'
})
geoplotlib.show()
