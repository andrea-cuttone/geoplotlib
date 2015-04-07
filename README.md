geoplotlib is a python toolbox for visualizing geographical data and making maps

# A first example
To produce a dot density map:

```python
data = read_csv('data/bus.csv')
geoplotlib.dot(data)
geoplotlib.show()
```

This will launch the geoplotlib window and plot the points on OpenStreetMap tiles, also allowing zooming and panning. geoplotlib automatically handles the data loading, the map projection, downloading the map tiles and the graphics rendering with OpenGL.

![demo](http://i.imgur.com/hr9GnLE.gif)

# Examples gallery

_Dot Density_
![dot](https://raw.githubusercontent.com/andrea-cuttone/geoplotlib/master/examples/screenshots/dotdensity.png)

_Heatmap (Kernel Density estimation)_
![kde](https://raw.githubusercontent.com/andrea-cuttone/geoplotlib/master/examples/screenshots/kde1.png)

_Shapefiles_
![shapefiles](https://raw.githubusercontent.com/andrea-cuttone/geoplotlib/master/examples/screenshots/shapefiles.png)

_Voronoi tessellation_
![voronoi](https://raw.githubusercontent.com/andrea-cuttone/geoplotlib/master/examples/screenshots/voronoi-filled.png)

_Spatial graph_
![graph](https://raw.githubusercontent.com/andrea-cuttone/geoplotlib/master/examples/screenshots/graph-flights.png)

All examples including source code are [here](https://github.com/andrea-cuttone/geoplotlib/tree/master/examples)

# Installation

geoplotlib requires:
* [numpy](http://www.numpy.org/)
* [pyglet](http://www.pyglet.org/)
	* **note:** in order for pyglet to work with ipython on Mac, [this workaround](https://code.google.com/p/pyglet/issues/detail?id=728) is needed. [Here](https://code.google.com/r/andreacuttone-pyglet-multipleruns/) you can find a fork of the pyglet dev branch with the workaround applied

optional requirements:
* [matplotlib](http://matplotlib.org/) for colormaps
* [scipy](http://www.scipy.org) for some layers
* [pyshp](https://github.com/GeospatialPython/pyshp) for reading .shp files

to install, run:

```python setup.py install```

# User Guide
A detailed user guide can be found in the [wiki](https://github.com/andrea-cuttone/geoplotlib/wiki/User-Guide)
