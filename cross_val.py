import numpy as np
import torch
import scipy.io as sio
import h5py
from sklearn.model_selection import StratifiedKFold, ShuffleSplit, GroupShuffleSplit
import importlib

class GraphDataset(torch.utils.data.Dataset):
    ''' Sample graphs and nodes in graph
    '''
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        var = ['static_R', 'label', 'subject_id']
        feature, label, subject_id = Eloadmat(self.files[idx], var)
        label[label==-1]=0
        feature = triu2mat(feature)
        return feature, label, subject_id

class fcDataset(torch.utils.data.Dataset):

    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        var = ['static_R', 'label', 'subject_id']
        res = Eloadmat(self.files[idx], var)
        return res

def triu2mat(trilfeat):

    N, N2 = trilfeat.shape
    dim = int((1+np.sqrt(1+8*N2))/2)
    index = np.ones((dim,dim))
    index = np.triu(index,1)

    matrix = np.zeros((N,dim,dim),dtype=trilfeat.dtype)
    for i in range(N):
        temp = np.zeros((dim,dim),dtype=trilfeat.dtype)
        temp[index==1]=trilfeat[i,:]
        temp = temp + temp.T
        matrix[i,:,:] = temp
    return matrix

def Eloadmat(datafile, var):
    if isinstance(datafile, list):
        res = [[]] * len(var)
        for i, file in enumerate(datafile):
            tmpres = Eloadmat(file, var)
            for j, tmp in enumerate(tmpres):
                res[j].extend(tmp)

        for i, tmp in enumerate(res):
            res[i] = np.stack(tmp, 0)
        return res

    res = []
    try:
        ## read normal .mat
        data = sio.loadmat(datafile)
        for tmp in var:
            if tmp not in data.keys():
                res.append([])
                continue
            res.append(data[tmp])
    except:
        ## read -v7.3 .mat
        data = h5py.File(datafile, 'r')
        for tmp in var:
            if tmp not in data.keys():
                res.append([])
                continue
            X = np.transpose(data[tmp])
            if X.dtype == np.dtype('uint16'):
                X = bytes(X[:]).decode('utf-16')
            res.append(X)
        data.close()
    return res


def split_kfold(n_splits=10, index=None, label=None, shuffle=False):
    if label is None:
        label = np.ones(len(index))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle)
    return list(skf.split(index, label))


def split_train_val_test(n_splits=10, index=None, label=None, shuffle=False):
    if label is None:
        label = np.ones(len(index))
    if not isinstance(n_splits, list):
        n_splits = [n_splits] * 2
    skf = StratifiedKFold(n_splits=n_splits[0], shuffle=shuffle)
    skf2 = StratifiedKFold(n_splits=n_splits[1], shuffle=False)
    all_idx = []
    for train_idx, test_idx in skf.split(index, label):
        tmpidx1, tmpidx2 = list(skf2.split(train_idx, np.array(label)[train_idx]))[0]
        all_idx.append((train_idx[tmpidx1], train_idx[tmpidx2], test_idx))

    return all_idx


def split_train_val_test_balance(val_test_num=100, index=None, label=None, shuffle=False):
    if label is None:
        label = np.ones(len(index))
    if not isinstance(val_test_num, list):
        val_test_num = [val_test_num] * 2

    mlb = int(max(label))
    sf = ShuffleSplit(n_splits=1, test_size=val_test_num[0], random_state=7)
    sf2 = ShuffleSplit(n_splits=1, test_size=val_test_num[1], random_state=7)

    train_idxs = []
    val_idxs = []
    test_idxs = []
    for i in range(mlb + 1):
        idx = np.nonzero(label == i)[0]
        tmpsub = index[idx]
        train_idx_tmp, test_idx = list(sf.split(tmpsub))[0]
        tmpidx1, tmpidx2 = list(sf2.split(tmpsub[train_idx_tmp]))[0]
        train_idx = idx[train_idx_tmp[tmpidx1]]
        val_idx = idx[train_idx_tmp[tmpidx2]]
        test_idx = idx[test_idx]
        train_idxs.append(train_idx)
        val_idxs.append(val_idx)
        test_idxs.append(test_idx)
    train_idxs = np.concatenate(train_idxs)
    val_idxs = np.concatenate(val_idxs)
    test_idxs = np.concatenate(test_idxs)
    return (train_idxs, val_idxs, test_idxs)


def apply_splits(files, index, subject=None, uniq_subject=None):
    if not (isinstance(index[0], list) or isinstance(index[0], np.ndarray)):
        index = [index]
    out_files = []
    for idx in index:
        if subject != None:
            sub = np.array(uniq_subject)[idx]
            idx = np.isin(np.array(subject), sub)
        test_files = np.array(files)[idx].tolist()
        out_files.append(test_files)

    return out_files


def file2loader(files, batch_size, shuffle=True, num_workers=0, sampler=None, labels=None):
    if not (isinstance(files[0], list) or isinstance(files[0], np.ndarray)):
        files = [files]
    if not isinstance(batch_size, list):
        batch_size = [batch_size] * len(files)
    if not isinstance(shuffle, list):
        if len(files) == 3:
            shuffle = [shuffle, False, False]
        else:
            shuffle = [shuffle, False]

    dataset_loader = []
    for i, file in enumerate(files):
        if i == 0 and sampler:
            flag = False
            sampler_dic = get_sampler(sampler)
            sampler = sampler_dic['sampler'](labels, **sampler_dic['params'])
        else:
            sampler = None
            flag = True
        dataset_sampler = GraphDataset(file)
        tmp_loader = torch.utils.data.DataLoader(
            dataset_sampler, batch_size=batch_size[i],
            shuffle=shuffle[i] & flag,
            sampler=sampler,
            num_workers=num_workers)
        dataset_loader.append(tmp_loader)

    return dataset_loader


def cross_val_loader(files, subject, uniq_subject, batch_size, ith_fold, label=None, n_splits=10, sampler=None):
    if subject != None:
        index = uniq_subject
    else:
        index = files
    split_index = split_train_val_test(n_splits=n_splits, index=index, label=label)
    split_files = apply_splits(files=files, index=split_index[ith_fold], subject=subject, uniq_subject=uniq_subject)
    if sampler:
        labels = Eloadmat(split_files[0], ['label'])[0]
    else:
        labels = None
    split_loader = file2loader(files=split_files, batch_size=batch_size, shuffle=True, num_workers=8, sampler=sampler,
                               labels=labels)

    return split_loader

def cross_train_test_loader(files, subject, uniq_subject, batch_size, ith_fold, label=None, n_splits=10, sampler=None):
    if subject != None:
        index = uniq_subject
    else:
        index = files
    split_index = split_kfold(n_splits=n_splits, index=index, label=label)
    split_files = apply_splits(files=files, index=split_index[ith_fold], subject=subject, uniq_subject=uniq_subject)
    if sampler:
        labels = Eloadmat(split_files[0], ['label'])[0]
    else:
        labels = None
    split_loader = file2loader(files=split_files, batch_size=batch_size, shuffle=True, num_workers=8, sampler=sampler,
                               labels=labels)

    return split_loader


def train_val_test_loader(files, subject, uniq_subject, batch_size, label=None, val_test_num=100, sampler=None):
    if subject != None:
        index = uniq_subject
    else:
        index = files
    split_index = split_train_val_test_balance(val_test_num=val_test_num, index=index, label=label)
    split_files = apply_splits(files=files, index=split_index, subject=subject, uniq_subject=uniq_subject)
    if sampler:
        labels = Eloadmat(split_files[0], ['label'])[0]
    else:
        labels = None
    split_loader = file2loader(files=split_files, batch_size=batch_size, shuffle=True, num_workers=8, sampler=sampler,
                               labels=labels)

    return split_loader


def source_import(file_path):
    """This function imports python module directly from source code using importlib"""
    spec = importlib.util.spec_from_file_location('', file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_sampler(types):
    if types == 'ClassAwareSampler':
        sampler_defs = {'def_file': 'ClassAwareSampler.py', 'num_samples_cls': 4, 'type': 'ClassAwareSampler'}
    else:
        sampler_defs = {'alpha': 1.0, 'cycle': 0, 'decay_gap': 30, 'def_file': 'MixedPrioritizedSampler.py',
                        'epochs': 90, 'fixed_scale': 1, 'lam': 1.0, 'manual_only': True, 'nroot': 2.0, 'ptype': 'score',
                        'rescale': False, 'root_decay': None, 'type': 'MixedPrioritizedSampler'}

    if sampler_defs:
        if sampler_defs['type'] == 'ClassAwareSampler':
            sampler_dic = {
                'sampler': source_import(sampler_defs['def_file']).get_sampler(),
                'params': {'num_samples_cls': sampler_defs['num_samples_cls']}
            }
        elif sampler_defs['type'] in ['MixedPrioritizedSampler',
                                      'ClassPrioritySampler']:
            sampler_dic = {
                'sampler': source_import(sampler_defs['def_file']).get_sampler(),
                'params': {k: v for k, v in sampler_defs.items() \
                           if k not in ['type', 'def_file']}
            }
    else:
        sampler_dic = None
    return sampler_dic