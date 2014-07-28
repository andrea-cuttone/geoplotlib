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
