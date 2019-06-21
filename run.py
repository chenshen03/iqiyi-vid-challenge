# -*- coding: utf-8 -*-

import os
import shutil
import sys
import pickle
import numpy as np


PARTS = 4
AUG = 0

DET_MODEL='./model/model-r50-gg/model,0'
MODEL='./model/model-r100-gg/model,0' 

OUTPUT_DIR='./iqiyi_vid/data2/det_results_1904'

DET_PREFIX='det_trainval'
TEST_DET_PREFIX='det_trainval_test'
FEAT_PREFIX='all_feat_trainval'
TEST_FEAT_PREFIX='testfeat'

MODE = int(sys.argv[1])

# 检测出视频中所有人脸，并根据特征norm去除掉质量较低的人脸
if MODE==1:
  for i in range(PARTS):
    det_file=os.path.join(OUTPUT_DIR, '%s%d%d'%(DET_PREFIX, i, PARTS))
    cmd = "python2 -u detect.py --model %s --output %s --stage trainval --gpu %d --split %d,%d > logs/d%d.log 2>&1 & " % (DET_MODEL, det_file, i%4, i, PARTS, i)
    if (i < 4):
        print(cmd)
        os.system(cmd)

        
# 将人脸图像转换为512维特征向量，每个视频对应一个向量，序列化存储
elif MODE==2:
  for i in range(PARTS):
    if (i < 4):
        feat_file=os.path.join(OUTPUT_DIR, '%s%d%d'%(FEAT_PREFIX, i, PARTS))
        assert not os.path.exists(feat_file)
        # if os.path.exists(feat_file):
        #   os.remove(feat_file)
        det_file=os.path.join(OUTPUT_DIR, '%s%d%d'%(DET_PREFIX, i, PARTS))
        assert os.path.exists(det_file)
        cmd="python2 -u feature.py --model %s --input %s --output %s --gpu %d --sampling 3 --aug %d > logs/all_f%d.log 2>&1 & "%(MODEL, det_file, feat_file, i%4, AUG, i)
        print(cmd)
        os.system(cmd)

        
# 将MODE2得到的train data和val data再处理保存
elif MODE==3:
  FPS = ['feat_trainval']
  inputs = []
  for fp in FPS:
    for i in range(PARTS):
      inputs.append(os.path.join(OUTPUT_DIR,'%s%d%d'%(fp,i,PARTS)))
  inputs = ','.join(inputs)
  output_file=os.path.join(OUTPUT_DIR, 'trainval')
  if os.path.exists(output_file):
    os.remove(output_file)
  cmd="python2 genfeat.py --inputs %s --output %s" % (inputs, output_file)
  print(cmd)
  # os.system(cmd)


# 将MODE2得到的val/test data再处理保存
elif MODE==4:
  FPS = ['feat_trainval']
  inputs = []
  for fp in FPS:
    for i in range(PARTS):
      inputs.append(os.path.join(OUTPUT_DIR,'%s%d%d'%(fp,i,PARTS)))
  inputs = ','.join(inputs)
  output_file=os.path.join(OUTPUT_DIR, 'val')
  if os.path.exists(output_file):
    os.remove(output_file)
  cmd="python2 genfeat_val.py --inputs %s --output %s" % (inputs, output_file)
  print(cmd)
  # os.system(cmd)


# DEBUG模式
elif MODE==99:
  INPUT_DIR = './iqiyi_vid/data2'
  OUTPUT_DIR = './iqiyi_vid/top5'
  FEAT_PREFIX='feat_trainval'
  for i in range(PARTS):
    if (i >= 4):
        feat_file=os.path.join(OUTPUT_DIR, '%s%d%d'%(FEAT_PREFIX, i, PARTS))
        if os.path.exists(feat_file):
          os.remove(feat_file)
        det_file=os.path.join(INPUT_DIR, '%s%d%d'%(DET_PREFIX, i, PARTS))
        assert os.path.exists(det_file)
        cmd="python2 -u feature_debug.py --model %s --input %s --output %s --gpu %d --sampling 3 --aug %d > ./logs/top5_f%d.log 2>&1 & "%(MODEL, det_file, feat_file, i%4, AUG, i)
        print(cmd)
        os.system(cmd)