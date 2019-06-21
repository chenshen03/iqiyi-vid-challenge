
import os
import pickle
import argparse
import random
import numpy as np
import sklearn
import sklearn.preprocessing

parser = argparse.ArgumentParser(description='face model test')
parser.add_argument('--inputs', default='./iqiyi_vid/data2/feat_testa', help='')
parser.add_argument('--output', default='./iqiyi_vid/data2/feat_vala', help='')
args = parser.parse_args()

PARTS = [1, 2, 3]
MAX_LABEL = 99999
MODE = 2

print(args, MAX_LABEL, PARTS, MODE)

inputs = args.inputs.split(',')

streams = []
for input in inputs:
  filename = input
  assert os.path.exists(filename)
  f = open(filename, 'rb')
  streams.append(f)

fout = open(args.output, 'wb')

cnt = 0
for f in streams:
  while True:
    try:
      item = pickle.load(f)
    except:
      break
    name = item[0]
    feat = item[1]
    label = item[2]
    flag = item[3]
    if flag == 2 and label != -1:
        if np.isnan(feat.max()) or np.isnan(feat.min()):
            print name, label, flag
        else:
            cnt += 1
            pickle.dump(item, fout, protocol=pickle.HIGHEST_PROTOCOL)

print cnt
        
for f in streams:
  f.close()

fout.close()

