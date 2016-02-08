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

[Dot Density Map](https://github.com/andrea-cuttone/geoplotlib/blob/master/examples/dot.py)
![dot](https://raw.githubusercontent.com/andrea-cuttone/geoplotlib/master/examples/screenshots/dotdensity.png)

[Heatmap (Kernel Density estimation)](https://github.com/andrea-cuttone/geoplotlib/blob/master/examples/kde.py)
![kde](https://raw.githubusercontent.com/andrea-cuttone/geoplotlib/master/examples/screenshots/kde1.png)

[Shapefiles](https://github.com/andrea-cuttone/geoplotlib/blob/master/examples/shapefiles.py)
![shapefiles](https://raw.githubusercontent.com/andrea-cuttone/geoplotlib/master/examples/screenshots/shapefiles.png)

[Choropleth and GeoJSON](https://github.com/andrea-cuttone/geoplotlib/blob/master/examples/choropleth.py)
![choropleth](https://raw.githubusercontent.com/andrea-cuttone/geoplotlib/master/examples/screenshots/choropleth.png)

[Voronoi tessellation](https://github.com/andrea-cuttone/geoplotlib/blob/master/examples/voronoi-filled.py)
![voronoi](https://raw.githubusercontent.com/andrea-cuttone/geoplotlib/master/examples/screenshots/voronoi-filled.png)

[Spatial graph](https://github.com/andrea-cuttone/geoplotlib/blob/master/examples/graph-flights.py)
![graph](https://raw.githubusercontent.com/andrea-cuttone/geoplotlib/master/examples/screenshots/graph-flights.png)

All examples source code is [here](https://github.com/andrea-cuttone/geoplotlib/tree/master/examples)

# Installation

geoplotlib requires:
* [numpy](http://www.numpy.org/)
* [pyglet 1.2.4](http://www.pyglet.org/download.html)
	* **note:** in order for pyglet to work with ipython on Mac, version 1.2.4 or newer is needed

optional requirements:
* [matplotlib](http://matplotlib.org/) for colormaps
* [scipy](http://www.scipy.org) for some layers
* [pyshp](https://github.com/GeospatialPython/pyshp) for reading .shp files

to install from source run:

```python setup.py install```

or with pip:

```pip install geoplotlib```

# User Guide
A detailed user guide can be found in the [wiki](https://github.com/andrea-cuttone/geoplotlib/wiki/User-Guide)
