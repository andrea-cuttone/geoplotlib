geoplotlib is a python toolbox for visualizing geographical data and making maps

# A first example
To produce a scatterplot:

```python
data = read_csv('data/bus.csv')
geoplotlib.scatter(data)
geoplotlib.show()
```

![scatter](https://raw.githubusercontent.com/andrea-cuttone/geoplotlib/master/examples/screenshots/scatter.png)

# Examples gallery

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
