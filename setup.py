from setuptools import setup

setup(name='geoplotlib',
      use_2to3=True,
      packages=['geoplotlib'],
      version='0.3.0',
      install_requires=[
    	'numpy>=1.7.0',
        'pyglet>=1.2.0',
        'pyshp',
        'scipy>=0.12.0',
        'matplotlib',
      ],
      description='python toolbox for geographic visualizations',
      author='Andrea Cuttone',
      author_email='andreacuttone@gmail.com',
      url='https://github.com/andrea-cuttone/geoplotlib',
)
