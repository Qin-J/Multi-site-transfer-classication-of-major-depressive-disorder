#!/usr/bin/env bash

# copy results file to a new directory

datapath='/home/nudt/qinjian/graph_convolution/data/multicentors_data_sub_normalized_MDD_meta_and_ours/MDD_allroi'
pretrain_path='pretrain_model_multi-site_transfer_learning_allroi'

python train_fmridata_transfer_leave_site_out_allAtlas_MDD.py --method=GCNSP --train_or_test=test --datadir=${datapath} --pretrain_dir=${pretrain_path} --cuda=0   
