# Multi-site-transfer-classification-of-major-depressive-disorder
The core code for article "Multi-site-transfer-classification-of-major-depressive-disorder"

## System Requirements

>### Software requirements

The package has been tested on the Ubuntu 18.04, Python 3.6 and Matlab 2009

>### Python Dependencies
This project mainly depends on the following Python stack: <br>

pytorch 1.4.0 <br>
numpy <br>
sklearn <br>
scipy <br>
h5py <br>
argparse <br>
os <br>
time <br>
warnings <br>

## Usage
>### 1. For GCN and GCNSP models:
>>#### 1.1 For multi-site pooling classification, please run in Linux terminal:

```python train_fmridata_MDD_simple.py --method=GCNSP --train_or_test=train --datadir=${datapath} --pretrain_dir=${pretrain_path} --cuda=0```

where, --method denotes the used model (GCN or GCNSP). --train_or_test denotes that training from scratch, or only testing based on our trained models.
--datadir is the directory where functional connectivity data is located. --pretain_dir is the directory where trained model is located'. --cuda denotes the GPU to be used.

For an example of test mode, please run in Linux terminal: <br> 
```demo_multi_site_pooling_classification.sh```

>>#### 1.2 For For multi-site transfer learning classification, please run in Linux terminal:

```python train_fmridata_transfer_leave_site_out_allAtlas_MDD.py --method=GCNSP --train_or_test=train --datadir=${datapath} --pretrain_dir=${pretrain_path} --cuda=0``` 

For an example of test mode, please run in Linux terminal: <br>
```demo_multi_site_transfer_classification.sh```

>### 2. For RFE-LDA, RFE-LR, and RFE-SVM models:

>>#### 2.1 For multi-site pooling classification, please run in Matlab:

```./LDA_LR_SVM_models_matlab_code/demo_multisite_pooling_classification_multimodels.m```


>>#### 2.2 For single site classification via RFE-SVM, please run in Matlab:

```./LDA_LR_SVM_models_matlab_code/demo_single_site_classification_RFE_SVM.m```


Note that: For the above scripts, to get the results of our paper, you need to download the data (https://pan.baidu.com/s/112X8Ogs4oDJnU8lxcb3pWg, with the extraction code of x3sv), then decompress it, then set the datapath in the scripts, and finally run the scripts.

## License
This project is covered under the Apache 2.0 License.
