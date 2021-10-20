from setuptools import setup

setup(name='geoplotlib',
      use_2to3=False,
      packages=['geoplotlib'],
      version='0.3.2',
      description='python toolbox for geographic visualizations',
      author='Andrea Cuttone',
      author_email='andreacuttone@gmail.com',
      url='https://github.com/andrea-cuttone/geoplotlib',
      install_requires=['numpy>=1.12','pyglet>=1.2.4']
)
