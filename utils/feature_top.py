import os
import pickle
import argparse
import random
import numpy as np


parser = argparse.ArgumentParser(description='face model test')
parser.add_argument('--input', default='./iqiyi_vid/det_results_1904/all_feat_trainval04', help='')
parser.add_argument('--output', default='feat_trainval04', help='')
parser.add_argument('--output_dir', default='./iqiyi_vid/data3', help='')
parser.add_argument('--top', type=int, default=10, help='')
args = parser.parse_args()
print(args)


OUTPUT_DIR = os.path.join(args.output_dir, 'top'+str(args.top))
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print "create dir:", OUTPUT_DIR

output_file = os.path.join(OUTPUT_DIR, args.output)


fin = open(args.input, 'rb')
fout = open(output_file, 'wb')

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
    x_sort = np.array(sorted(x, key=lambda a: np.linalg.norm(a)))[::-1]
    feat = np.sum(x_sort[:args.top], axis=0)
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