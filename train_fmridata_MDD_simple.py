
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
import argparse
import os
import time
import warnings

import numpy as np
import sklearn.metrics as metrics
import scipy
import scipy.io as sio

import cross_val
import encoders
import graph

def evaluate(dataset, adj, model, args, name='Validation', max_num_examples=None):
    model.eval()

    labels = []
    preds_label = []
    preds_score = []
    subid = []
    avg_loss = 0.0
    for batch_idx, data in enumerate(dataset):

        h0 = Variable(torch.squeeze(data[0]).float(), requires_grad=False).cuda()
        label = Variable(torch.squeeze(data[1]).long()).cuda()
        labels.append(label.cpu().detach().numpy())

        subid.append(np.squeeze(np.array(data[2])))

        if len(h0.size()) < 3:
            h0.unsqueeze_(0)

        ypred, _ = model(h0, adj)
        _, indices = torch.max(ypred, 1)
        preds_label.append(indices.cpu().data.numpy())
        avg_loss += model.loss(ypred, label).cpu().detach().numpy()
        preds_score.append(ypred.cpu().data.numpy())

        if max_num_examples is not None:
            if (batch_idx + 1) * args.batch_size > max_num_examples:
                break

    avg_loss /= batch_idx + 1
    labels = np.hstack(labels)
    preds_label = np.hstack(preds_label)

    preds_score = np.concatenate(preds_score, 0)

    subid = np.hstack(subid)
    uniq_subid, ind1, ind2 = np.unique(subid, return_index=True, return_inverse=True)
    uniq_preds = []
    for idx, sub in enumerate(uniq_subid):
        tempind = np.isin(ind2, idx)
        uniq_preds.append(preds_score[tempind, :].mean(axis=0))

    uniq_preds = np.vstack(uniq_preds)
    uniq_preds_label = np.argmax(uniq_preds, 1)
    uniq_label = labels[ind1]

    result = {'prec': metrics.precision_score(labels, preds_label, average='macro'),
              'recall': metrics.recall_score(labels, preds_label, average='macro'),
              'acc': metrics.accuracy_score(labels, preds_label),
              'merge_acc': metrics.accuracy_score(uniq_label, uniq_preds_label),
              'F1': metrics.f1_score(labels, preds_label, average="micro"),
              'loss': avg_loss,
              'subjectid': subid,
              'pred_scores': preds_score,
              'labels': labels}
    print(name, " accuracy:", result['acc'], "merge accuracy:", result['merge_acc'])
    return result


def train(dataset, data_adj, model, args, same_feat=True, val_dataset=None, test_dataset=None, writer=None,
          mask_nodes=True):

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4) # wd = 1e-4
    scheduler = MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)
    iter = 0
    best_val_result = {
        'epoch': 0,
        'loss': 0,
        'acc': 0}
    test_result = {
        'epoch': 0,
        'loss': 0,
        'acc': 0}
    train_accs = []
    train_epochs = []
    best_val_accs = []
    best_val_epochs = []
    test_accs = []
    test_epochs = []
    val_accs = []
    val_merge_accs = []
    val_pred = []
    val_subid = []
    val_labels = []

    train_loss = []
    val_loss = []


    # csr matrix to torch.sparse
    data_adj = data_adj.tocoo()
    indices = np.vstack((data_adj.row, data_adj.col))
    values = data_adj.data
    shape = data_adj.shape

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    adj = Variable(torch.sparse.FloatTensor(i, v, torch.Size(shape)), requires_grad=False).cuda()

    for epoch in range(args.num_epochs):  # args.num_epochs
        begin_time = time.time()
        avg_loss = 0.0
        model.train()
        scheduler.step()
        optimizer.zero_grad()
        print('Epoch: ', epoch)
        for batch_idx, data in enumerate(dataset):
            model.zero_grad()

            h0 = Variable(torch.squeeze(data[0]).float(), requires_grad=False).cuda()
            label = Variable(torch.squeeze(data[1]).long()).cuda()

            if len(h0.size()) < 3:
                h0.unsqueeze_(0)

            if len(label.size()) < 1:
                continue
                label.unsqueeze_(0)

            ypred, _ = model(h0, adj)
            loss = model.loss(ypred, label)

            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), args.clip)
            optimizer.step()
            iter += 1
            avg_loss += loss

        avg_loss /= batch_idx + 1
        elapsed = time.time() - begin_time

        print('Avg loss: ', avg_loss, '; epoch time: ', elapsed)
        result = evaluate(dataset, adj, model, args, name='Train')
        train_accs.append(result['acc'])
        train_loss.append(result['loss'])
        train_epochs.append(epoch)
        if val_dataset is not None:
            val_result = evaluate(val_dataset, adj, model, args, name='Validation')
            val_accs.append(val_result['acc'])
            val_merge_accs.append(val_result['merge_acc'])
            val_pred.append(val_result['pred_scores'])
            val_loss.append(val_result['loss'])
            if epoch==0:
                val_subid.append(val_result['subjectid'])
                val_labels.append(val_result['labels'])

        if val_result['acc'] > best_val_result['acc'] - 1e-7:
            best_val_result['acc'] = val_result['acc']
            best_val_result['epoch'] = epoch
            best_val_result['loss'] = avg_loss
            best_val_result['merge_acc'] = val_result['merge_acc']

        if test_dataset is not None:
            test_result = evaluate(test_dataset, adj, model, args, name='Test')
            test_result['epoch'] = epoch

        print('Best val result: ', best_val_result)
        best_val_epochs.append(best_val_result['epoch'])
        best_val_accs.append(best_val_result['acc'])
        if test_dataset is not None:
            print('Test result: ', test_result)
            test_epochs.append(test_result['epoch'])
            test_accs.append(test_result['acc'])

    return model, val_accs, val_merge_accs, val_pred, val_subid, val_labels, val_loss, train_accs, train_loss


def arg_parse():
    parser = argparse.ArgumentParser(description='GraphPool arguments.')

    parser.add_argument('--datadir', dest='datadir',
                        help='Directory where functional connectivity data is located')
    parser.add_argument('--pretrain_dir', dest='pretrain_dir',
                        help='Directory where pretrained model is located')
    parser.add_argument('--logdir', dest='logdir',
                        help='Tensorboard log directory')
    parser.add_argument('--cuda', dest='cuda',
                        help='CUDA.')
    parser.add_argument('--max-nodes', dest='max_nodes', type=int,
                        help='Maximum number of nodes (ignore graghs with nodes exceeding the number.')
    parser.add_argument('--lr', dest='lr', type=float,
                        help='Learning rate.')
    parser.add_argument('--clip', dest='clip', type=float,
                        help='Gradient clipping.')
    parser.add_argument('--batch-size', dest='batch_size', type=int,
                        help='Batch size.')
    parser.add_argument('--epochs', dest='num_epochs', type=int,
                        help='Number of epochs to train.')
    parser.add_argument('--train-ratio', dest='train_ratio', type=float,
                        help='Ratio of number of graphs training set to all graphs.')
    parser.add_argument('--num_workers', dest='num_workers', type=int,
                        help='Number of workers to load data.')
    parser.add_argument('--feature', dest='feature_type',
                        help='Feature used for encoder. Can be: id, deg')
    parser.add_argument('--input-dim', dest='input_dim', type=int,
                        help='Input feature dimension')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int,
                        help='Hidden dimension')
    parser.add_argument('--output-dim', dest='output_dim', type=int,
                        help='Output dimension')
    parser.add_argument('--num-classes', dest='num_classes', type=int,
                        help='Number of label classes')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int,
                        help='Number of graph convolution layers before each pooling')
    parser.add_argument('--nobn', dest='bn', action='store_const',
                        const=False, default=True,
                        help='Whether batch normalization is used')
    parser.add_argument('--dropout', dest='dropout', type=float,
                        help='Dropout rate.')
    parser.add_argument('--nobias', dest='bias', action='store_const',
                        const=False, default=True,
                        help='Whether to add bias. Default to True.')

    parser.add_argument('--method', dest='method',
                        help='Method. Possible values: base, base-set2set, soft-assign')
    parser.add_argument('--name-suffix', dest='name_suffix',
                        help='suffix added to the output filename')

    parser.add_argument('--train_or_test', dest='train_or_test',
                        help='train or test mode')

    parser.set_defaults(datadir='MDD_allroi',
                        pretrain_dir='pretrain_model_multi-site_pooling',
                        logdir='log',
                        dataset='syn1v2',
                        max_nodes=1000,
                        cuda='2',
                        feature_type='default',
                        lr=0.0001,
                        clip=2.0,
                        batch_size=32,
                        num_epochs=100,
                        train_ratio=0.8,
                        test_ratio=0.1,
                        num_workers=1,
                        input_dim=10,
                        hidden_dim=64,
                        output_dim=64,
                        num_classes=2,
                        num_gc_layers=2,
                        dropout=0.5,
                        method='GCN',  # 'GCN'  'GCNSP'
                        name_suffix='',
                        assign_ratio=0.25,
                        num_pool=1,
                        train_or_test='train'  # 'train'  'test'
                        )
    return parser.parse_args()


# main function
args = arg_parse()
writer = None
warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
print('CUDA', args.cuda)

tmpdata = scipy.io.loadmat('select_subject_final.mat')
selectSub = tmpdata['final_subject_ids']
# args.datadir = '/home/nudt/qinjian/graph_convolution/data/multicentors_data_sub_normalized_MDD_meta_and_ours/MDD_allroi'
datapath = args.datadir
folders = os.listdir(datapath)
folders = sorted(folders)

sites_acc = []
sites_merge_acc = []
all_merge = []
all_vals = []
all_val_pred = []
all_val_subid = []
all_val_labels = []

all_val_loss = []
all_train_acc = []
all_train_loss = []

files = []
subject_id = []
for fd in folders:
    filenames = os.listdir(datapath + '/' + fd)
    filenames = sorted(filenames)
    for f in filenames:
        subject_id.append(fd.split('_split')[0] + '_' + f.split('_run')[0])
        # subject_id.append(fd.split('_')[0] + f.split('_run')[0])
        files.append(datapath + '/' + fd + '/' + f)


ind = np.isin(subject_id, selectSub)
subject_id = np.array(subject_id)[ind].tolist()
files = np.array(files)[ind].tolist()


seed = 777
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)


labels = []
for f in files:
    tmp = cross_val.Eloadmat(f, ['label'])
    labels.extend(tmp[0])
labels = np.stack(labels, 0)

uniq_subject_raw, uniq_index = np.unique(subject_id, return_index=True)
uniq_lables_raw = labels[uniq_index]

# setting training model
cross_val_model = True
save_model = not cross_val_model
finetune = False
SavePath = args.pretrain_dir
if not os.path.exists(SavePath):
    os.mkdir(SavePath)

roinum = 1720
input_dim = roinum
assign_input_dim = -1

# Get the graph structure
X = 0
for f in files:
    X += cross_val.Eloadmat(f, ['static_R'])[0]
X = X / len(files)
mean_connec = np.squeeze(cross_val.triu2mat(X))
dist, idx = graph.distance_lshforest(mean_connec, k=10, metric='cosine')
adj = graph.adjacency(dist, idx).astype(np.float32)
adj = graph.rescale_adj(adj)

# for batch nodes
max_num_nodes = adj.shape[0]
foldnum = 10

rind = np.random.permutation(len(uniq_subject_raw))
uniq_subject = uniq_subject_raw.copy()[rind]
uniq_labels = uniq_lables_raw.copy()[rind]

for i in range(foldnum):
    train_dataset, val_dataset = cross_val.cross_train_test_loader(
        files, subject_id, uniq_subject, args.batch_size, i, uniq_labels, foldnum)
    if args.method == 'GCN':
        model = encoders.GCNnet_orig(
            input_dim, args.hidden_dim, args.output_dim, args.num_classes,
            args.num_gc_layers, bn=args.bn, dropout=args.dropout, concat=False, args=args).cuda()
    else:
        model = encoders.GCNSPnet(
            input_dim, args.hidden_dim, args.output_dim, args.num_classes,
            args.num_gc_layers, bn=args.bn, dropout=args.dropout, concat=False, args=args).cuda()

    if finetune:
        model.load_state_dict(torch.load(SavePath + '/' + mdfile))

    if args.train_or_test=='train':
        _, val_accs, val_merge, val_pred, val_subid, val_labels, val_loss, train_accs, train_loss = train(train_dataset, adj, model, args, val_dataset=val_dataset,
                                          test_dataset=None, writer=writer)
        mdfile = 'trained_model_'+args.method+'_fold_'+str(i)+'.pt'
        torch.save(model.state_dict(), SavePath + '/' + mdfile)
    else:
        mdfile = 'trained_model_' + args.method + '_fold_' + str(i) + '.pt'
        model.load_state_dict(torch.load(SavePath + '/' + mdfile))

        if 'data_adj' not in vars():
            # csr matrix to torch.sparse
            data_adj = adj
            data_adj = data_adj.tocoo()
            indices = np.vstack((data_adj.row, data_adj.col))
            values = data_adj.data
            shape = data_adj.shape

            i = torch.LongTensor(indices)
            v = torch.FloatTensor(values)
            data_adj = Variable(torch.sparse.FloatTensor(i, v, torch.Size(shape)), requires_grad=False).cuda()

        val_result = evaluate(val_dataset, data_adj, model, args, name='Validation')
        val_accs = val_result['acc']
        val_merge = val_result['merge_acc']
        val_pred = val_result['pred_scores']
        val_loss = val_result['loss']
        val_subid = val_result['subjectid']
        val_labels = val_result['labels']
        train_accs = []
        train_loss = []

    all_vals.append(np.array(val_accs))
    all_merge.append(np.array(val_merge))
    all_val_pred.append(val_pred)
    all_val_subid.append(val_subid)
    all_val_labels.append(val_labels)

    all_val_loss.append(val_loss)
    all_train_acc.append(train_accs)
    all_train_loss.append(train_loss)

all_vals = np.vstack(all_vals)
all_val_loss = np.vstack(all_val_loss)
all_train_acc = np.vstack(all_train_acc)
all_train_loss = np.vstack(all_train_loss)
all_merge = np.hstack(all_merge)

if args.train_or_test=='train':
    print('final accuracy is %f , optimal accuracy is %f' % (all_vals.mean(axis=0)[-1], all_vals.mean(axis=0).max()))
else:
    print('final accuracy is %f ' % (all_vals.mean(axis=0)[-1]))

all_results = {'all_vals':all_vals,'all_merge':all_merge,'all_val_pred': all_val_pred, 'all_val_subid': all_val_subid, \
               'all_val_labels': all_val_labels,'all_val_loss':all_val_loss,'all_train_acc':all_train_acc,'all_train_loss':all_train_loss}
scipy.io.savemat('all_results_MDD_meta_'+args.method+'_fnn3_for_pred_largeROI_selectedSub_onlymeta_decreaseLR20_40_lr0001_wd0001_newcalc_666.mat', all_results)