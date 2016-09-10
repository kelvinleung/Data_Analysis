#!/usr/bin/env python
# coding=utf-8

import pandas as pd
import numpy as np
from pandas import DataFrame, Series
import matplotlib
import matplotlib.pyplot as plt
import json

file_name = 'data/usagov_bitly_data2012-03-16-1331923249.txt'
records = [json.loads(line) for line in open(file_name)]
time_zones = [rec['tz'] for rec in records if 'tz' in rec]
frame = DataFrame(records)

clean_tz = frame['tz'].fillna('Missing')
clean_tz[clean_tz == ''] = 'Unknown'
tz_counts = clean_tz.value_counts()

# UAs
results = Series([x.split()[0] for x in frame.a.dropna()])

cframe = frame[frame.a.notnull()]
operating_system = np.where(cframe['a'].str.contains('Windows'),
    'Windows', 'Not Windows')
by_tz_os = cframe.groupby(['tz', operating_system])
agg_counts = by_tz_os.size().unstack().fillna(0)
indexer =  agg_counts.sum(1).argsort()
count_subset = agg_counts.take(indexer)[-10:]
# make it percentage (sum = 1)
normed_subset = count_subset.div(count_subset.sum(1), axis=0)

# the hard way to count
def get_counts(sequence):
    counts = {}
    for x in sequence:
        if x in counts:
            counts[x] += 1
        else:
            counts[x] = 1
    return counts

def top_counts(count_dict, n = 10):
    value_key_pairs = [(count, tz) for tz, count in count_dict.items()]
    value_key_pairs.sort()
    return value_key_pairs[-n:]

if __name__ == '__main__':
    # print(top_counts(get_counts(time_zones)))
    print(tz_counts[:10])
    plt.figure()
    tz_counts[:10].plot(kind='barh', rot=0)
    normed_subset.plot(kind='barh', stacked=True)
    plt.show()
    # top 10 UA
    print(results.value_counts()[:10])
