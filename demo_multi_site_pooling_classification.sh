#!/usr/bin/env bash

# copy results file to a new directory

datapath='/home/nudt/qinjian/graph_convolution/data/multicentors_data_sub_normalized_MDD_meta_and_ours/MDD_allroi'
pretrain_path='pretrain_model_multi-site_pooling'

python train_fmridata_MDD_simple.py --method=GCN --train_or_test=test --datadir=${datapath} --pretrain_dir=${pretrain_path} --cuda=0   
