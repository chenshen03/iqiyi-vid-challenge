import pickle
import os


file1 = 'det_trainval04'
file2 = 'det_trainval14'
file3 = 'det_trainval24'
file4 = 'det_trainval34'
feat_name = 'det_trainval'

INPUT_DIR = '/media/disk1/chenshen/datasets/IQIYI_VID/data2/det_results_1904'
OUTPUT_DIR = '/media/disk1/chenshen/datasets/IQIYI_VID/data2/det_results_1904'

fin = open(os.path.join(INPUT_DIR, feat_name), 'rb')
fout1 = open(os.path.join(OUTPUT_DIR, file1), 'wb')
fout2 = open(os.path.join(OUTPUT_DIR, file2), 'wb')
fout3 = open(os.path.join(OUTPUT_DIR, file3), 'wb')
fout4 = open(os.path.join(OUTPUT_DIR, file4), 'wb')

vid = 0
while True:
    try:
        item = pickle.load(fin)
    except:
        break
    vid+=1
    name = item[0]
    feat = item[1]
    label = item[2]
    flag = item[3]
    
    if feat is None:
        continue
        
    namehash = hash(name)
    if namehash%4 == 0:
        pickle.dump(item, fout1, protocol=pickle.HIGHEST_PROTOCOL)
    elif namehash%4 == 1:
        pickle.dump(item, fout2, protocol=pickle.HIGHEST_PROTOCOL)
    elif namehash%4 == 2:
        pickle.dump(item, fout3, protocol=pickle.HIGHEST_PROTOCOL)
    elif namehash%4 == 3:
        pickle.dump(item, fout4, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(name, 'label', label, 'lenFeat', len(feat), 'flag', flag, 'vid', vid)
        
fin.close()
fout1.close()
fout2.close()
fout3.close()
fout4.close()