import os
import pickle
import argparse
import random
import numpy as np


parser = argparse.ArgumentParser(description='face model test')
parser.add_argument('--input', default='./iqiyi_vid/data2/det_results_1904/all_feat_trainval04', help='')
parser.add_argument('--output', default='./iqiyi_vid/data3/max_pooling/feat_trainval04', help='')
args = parser.parse_args()
print(args)


output_dir = os.path.dirname(args.output)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print "create dir:", output_dir

fin = open(args.input, 'rb')
fout = open(args.output, 'wb')

vid = 0
while True:
    try:
        item = pickle.load(fin)
    except:
        break
    vid += 1
    name = item[0]
    feats = item[1]
    label = item[2]
    flag = item[3]
    
    F = []
    x = feats[0]
    feat = x.max(axis=0)
    norm = np.linalg.norm(feat)
    feat /= norm
    F.append(feat)
    if len(F)==0:
        continue
    feat = np.array(F)
    
    print(name, 'label', label, 'lenR', len(feat), 'flag', flag, 'vid', vid)
    pickle.dump((name, feat, label, flag), fout, protocol=pickle.HIGHEST_PROTOCOL)
    
fout.close()
fin.close()