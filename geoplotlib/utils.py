from collections import defaultdict
import csv
from datetime import datetime


def read_csv(fname):
    values = defaultdict(list)
    with open(fname) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for (k,v) in row.items():
                try:
                    v = float(v)
                except:
                    try:
                        v = v.decode('utf-8')
                    except:
                        pass
                values[k].append(v)
    return {k: values[k] for k in values.keys()}


def epoch_to_str(epoch, fmt='%Y-%m-%d %H:%M:%S'):
    return datetime.fromtimestamp(epoch).strftime(fmt)
