import os
import shutil
import sys
import pickle
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='generate submit result')
# general
parser.add_argument('--submit-name', default='Submit_fbn_N100.txt', help='')
parser.add_argument('--output-dir', default='./iqiyi_vid/data3', help='')
parser.add_argument('--predict', default='pred_vala1', help='')
args = parser.parse_args()


#e>c>a>h>d
DEBUG='IQIYI_VID_TEST_0094168.mp4'
FPS = []
FPS.append(args.predict)
# for a in ['a']:
#     FPS.append('pred_vala_old%s'%(a))
weights = [1.0]*len(FPS)
size = len(FPS)
#size = 2
name2score = {}
inputs = []
dcount = 0
for i in range(size):
  fp = FPS[i]
  weight = weights[i]
  _input = os.path.join(args.output_dir, fp)
  print(_input, weight)
  f = open(_input, 'rb')
  while True:
    try:
      item = pickle.load(f)
    except:
      break
    #print(item)
    name = item[0]
    xscore = item[1]*weight
    #print(np.sum(xscore))
    #print(xscore)
    if name not in name2score:
      name2score[name] = xscore 
    else:
      #escore = name2score[name]
      #if escore[0]>xscore[0]:
      #  dcount+=1
      name2score[name] += xscore
    if name==DEBUG:
      print('debug found', np.sum(name2score[name]))
  f.close()
print(len(name2score), dcount)

ret_map = {}
TOPK = 100
N = 100
#S = 10000.0
S = 1.0
zcount = 0
for name, xscore in name2score.iteritems():
  if name==DEBUG:
    print('debug', name, np.sum(xscore))
  #else:
  #  print(name, np.sum(xscore))
  index = np.argsort(xscore)[::-1]
  index = index[:N]
  idfound = False
  idx = -1
  for im in index:
    idx+=1
    label = im
    score = xscore[im]
    if score<=0.0:
      break
    if idx==0 and label==0:
      zcount+=1
    #if idx<10:
    #  print(name, idx, label, score)
    if label==0:
      #if S==0 and idx==0:
      #  break
      continue
    #if idfound:
    if idx>0:
      score /= S
    if label not in ret_map:
      ret_map[label] = []
    ret_map[label].append( (name, score) )
    idfound = True
out_filename=os.path.join(args.output_dir, args.submit_name)
print(len(ret_map), zcount, out_filename)
outf = open(out_filename, 'w')
#out_filename2='./gsubmit_score.txt'
#outf2 = open(out_filename2, 'w')
empty_count=0
min_len = 99999
cnt = 0
for label, ret_list in ret_map.iteritems():
  ret_list = sorted(ret_list, key = lambda x : x[1], reverse=True)
  if TOPK>0 and len(ret_list)>TOPK:
    ret_list = ret_list[:TOPK]
  min_len = min(min_len, len(ret_list))
  out_items = [str(label)]
  #out_items2 = [str(label)]
  for ir, r in enumerate(ret_list):
    cnt += 1
    name = r[0]
    score = r[1]
    out_items.append(name)
    #out_items2.append('%.3f'%score)
  outf.write("%s\n"%(' '.join(out_items)))
  #outf2.write("%s\n"%(' '.join(out_items2)))
outf.close()
#outf2.close()
print('min', min_len)
print(cnt)