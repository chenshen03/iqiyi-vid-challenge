#!/usr/bin/env bash

# 1. Using pretrained model r100 to extract 512-dimentional features of images;
# 2. Try different way to fuse video features, including max-pooling, average-pooling, top K norm features and so on.

prefix=max_pooling
data_dir=./iqiyi_vid/data3/$prefix
submit_name=Submit_fbn_N100_$prefix.txt

python2 -u feature_debug.py --input ./iqiyi_vid/data2/det_results_1904/all_feat_trainval04 --output $data_dir/feat_trainval04

python2 -u feature_debug.py --input ./iqiyi_vid/data2/det_results_1904/all_feat_trainval14 --output $data_dir/feat_trainval14

python2 -u feature_debug.py --input ./iqiyi_vid/data2/det_results_1904/all_feat_trainval24 --output $data_dir/feat_trainval24

python2 -u feature_debug.py --input ./iqiyi_vid/data2/det_results_1904/all_feat_trainval34 --output $data_dir/feat_trainval34

python2 -u genfeat.py --inputs $data_dir/feat_trainval04,$data_dir/feat_trainval14,$data_dir/feat_trainval24,$data_dir/feat_trainval34 --output $data_dir/trainval

python2 -u genfeat_val.py --inputs $data_dir/feat_trainval04,$data_dir/feat_trainval14,$data_dir/feat_trainval24,$data_dir/feat_trainval34 --output $data_dir/val

python2 -u train_mlp.py --data $data_dir/trainval --prefix ./model/mlp/iqiyia2 --ce-loss --fbn

python2 -u predict.py --model ./model/mlp/iqiyia2,40 --inputs $data_dir/val --output $data_dir/pred_val2

python2 -u submit.py --predict pred_val2 --output-dir $data_dir --submit-name $submit_name

python2 -u evaluation_map_v3.py --gt-val ./iqiyi_vid/data1/gt_v2/val_v2.txt --my-val $data_dir/$submit_name


# 1. Using different pretrained model to extract 512-dimentional features of images and find out the influence of network on results.
# 2. The remaining steps follow the default settings.

prefix=r100_ar
data_dir=./iqiyi_vid/data3/$prefix
submit_name=Submit_fbn_N100_$prefix.txt

python2 -u feature.py --model ./model/model-r100-ar/model,1 --input ./iqiyi_vid/data2/det_results_1904/det_trainval04 --output $data_dir/feat_trainval04 --gpu 0 --sampling 3 --aug 0 > logs/retina_f0.log 2>&1 &

python2 -u feature.py --model ./model/model-r100-ar/model,1 --input ./iqiyi_vid/data2/det_results_1904/det_trainval14 --output $data_dir/feat_trainval14 --gpu 1 --sampling 3 --aug 0 > logs/retina_f1.log 2>&1 &

python2 -u feature.py --model ./model/model-r100-ar/model,1 --input ./iqiyi_vid/data2/det_results_1904/det_trainval24 --output $data_dir/feat_trainval24 --gpu 2 --sampling 3 --aug 0 > logs/retina_f2.log 2>&1 &

python2 -u feature.py --model ./model/model-r100-ar/model,1 --input ./iqiyi_vid/data2/det_results_1904/det_trainval34 --output $data_dir/feat_trainval34 --gpu 3 --sampling 3 --aug 0 > logs/retina_f3.log 2>&1 &

python2 -u genfeat.py --inputs $data_dir/feat_trainval04,$data_dir/feat_trainval14,$data_dir/feat_trainval24,$data_dir/feat_trainval34 --output $data_dir/trainval

python2 -u genfeat_val.py --inputs $data_dir/feat_trainval04,$data_dir/feat_trainval14,$data_dir/feat_trainval24,$data_dir/feat_trainval34 --output $data_dir/val

python2 -u train_mlp.py --data $data_dir/trainval --prefix ./model/mlp/iqiyia2 --ce-loss --fbn

python2 -u predict.py --model ./model/mlp/iqiyia2,40 --inputs $data_dir/val --output $data_dir/pred_val2

python2 -u submit.py --predict pred_val2 --output-dir $data_dir --submit-name $submit_name

python2 -u evaluation_map_v3.py --gt-val ./iqiyi_vid/data1/gt_v2/val_v2.txt --my-val $data_dir/$submit_name