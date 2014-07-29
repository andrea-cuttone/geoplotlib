from collections import defaultdict
import csv
from datetime import datetime


def read_csv(fname):
    values = defaultdict(list)
    with open(fname) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for (k,v) in row.items():
                values[k].append(parse_raw_value(v))
    return {k: values[k] for k in values.keys()}


def epoch_to_str(epoch, fmt='%Y-%m-%d %H:%M:%S'):
    return datetime.fromtimestamp(epoch).strftime(fmt)


def parse_raw_str(v):
    try:
        v = v.decode('utf-8')
    except:
        try:
            v = v.decode('latin1')
        except:
            pass
    return v


def parse_raw_value(v):
    try:
        v = float(v)
    except:
        v = parse_raw_str(v)
    return v


class BoundingBox():

    def __init__(self, north, west, south, east):
        self.north = north
        self.west = west
        self.south = south
        self.east = east


    @staticmethod
    def from_points(lons, lats):
        north, west = max(lats), min(lons)
        south, east = min(lats), max(lons)
        return BoundingBox(north=north, west=west, south=south, east=east)


    @staticmethod
    def from_bboxes(bboxes):
        north = max([b.north for b in bboxes])
        south = min([b.south for b in bboxes])
        west = min([b.west for b in bboxes])
        east = max([b.east for b in bboxes])
        return BoundingBox(north=north, west=west, south=south, east=east)

BoundingBox.WORLD = BoundingBox(north=85, west=-170, south=-85, east=190)
BoundingBox.DK = BoundingBox(north=57.769, west=7.932, south=54.444, east=12.843)
