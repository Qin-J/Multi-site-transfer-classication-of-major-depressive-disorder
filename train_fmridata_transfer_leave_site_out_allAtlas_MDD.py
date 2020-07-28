
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR

import argparse
import os
import time
import warnings

import scipy
import numpy as np
import sklearn.metrics as metrics

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
    # print(name, " accuracy:", result['acc'], "merge accuracy:", result['merge_acc'])
    return result

def train(dataset, data_adj, model, args, same_feat=True, val_dataset=None, test_dataset=None, writer=None,
          mask_nodes=True):

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4) #lr=0.0001, weight_decay=1e-4
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

    # csr matrix to torch.sparse
    data_adj = data_adj.tocoo()
    indices = np.vstack((data_adj.row, data_adj.col))
    values = data_adj.data
    shape = data_adj.shape

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    #    adj = Variable(torch.sparse.FloatTensor(i,v,torch.Size(shape)).to_dense(),requires_grad=False).cuda()
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

            batch_num_nodes = None

            if len(h0.size()) < 3:
                h0.unsqueeze_(0)

            if len(label.size()) < 1:
                label.unsqueeze_(0)
                continue

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
        # result = evaluate(dataset, adj, model, args, name='Train')
        # train_accs.append(result['acc'])
        # train_epochs.append(epoch)
        if val_dataset is not None:
            val_result = evaluate(val_dataset, adj, model, args, name='Validation')
            val_accs.append(val_result['acc'])
            val_pred.append(val_result['pred_scores'])
            if epoch==0:
                val_subid.append(val_result['subjectid'])
                val_labels.append(val_result['labels'])

        if val_result['acc'] > best_val_result['acc'] - 1e-7:
            best_val_result['acc'] = val_result['acc']
            best_val_result['epoch'] = epoch
            best_val_result['loss'] = avg_loss

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

    return model, val_accs, val_pred, val_subid, val_labels

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
                        pretrain_dir='pretrain_model_multi-site_transfer_learning_allroi',
                        logdir='log',
                        dataset='syn1v2',
                        max_nodes=1000,
                        cuda='3',
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
                        dropout=0.1,
                        method='GCNSP',  # 'GCN' 'GCNSP'
                        name_suffix='',
                        assign_ratio=0.25,
                        num_pool=1,
                        train_or_test='test'  # 'train'  'test'
                        )
    return parser.parse_args()


# main function
warnings.filterwarnings("ignore")
args = arg_parse()
writer = None
#args.cuda = '3'
cudaflag = str(int(args.cuda)+1)
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
print('CUDA', args.cuda)

tmpdata = scipy.io.loadmat('select_subject_final.mat')
selectSub = tmpdata['final_subject_ids']

# args.datadir = '/home/nudt/qinjian/graph_convolution/data/multicentors_data_sub_normalized_MDD_meta_and_ours/MDD_allroi'
datapath = args.datadir
folders_all = os.listdir(datapath)
folders_all = sorted(folders_all)

roinum = 1720
input_dim = roinum
assign_input_dim = -1

all_transfer_acc = []
all_baseline_acc = []
sample_ratio = []

all_transfer_pred = []
all_transfer_subid = []
all_transfer_labels = []
all_baseline_pred = []
all_baseline_subid = []
all_baseline_labels = []

all_tg_folder = []
runflage = 0
for ratio in range(len(folders_all)):
    runflage = runflage + 1
    print('++++++++++ sites %d ++++++++++' % (ratio))

    sub_folder = [folders_all[ratio]]
    folders = folders_all[:]
    folders.remove(sub_folder[0])

    files = []
    subject_id = []
    for fd in sub_folder:
        filenames = os.listdir(datapath + '/' + fd)
        filenames = sorted(filenames)
        for f in filenames:
            subject_id.append(fd.split('_split')[0] + '_' + f.split('_run')[0])
            files.append(datapath + '/' + fd + '/' + f)

    ind = np.isin(subject_id, selectSub)
    tg_subject_id = np.array(subject_id)[ind].tolist()
    tg_files = np.array(files)[ind].tolist()

    # tg_subject_id = subject_id
    # tg_files = files

    if len(tg_files) < 10:
        continue

    labels = []
    for f in tg_files:
        tmp = cross_val.Eloadmat(f, ['label'])
        labels.extend(tmp[0])
    labels = np.stack(labels, 0)
    if np.sum(labels<0)<20 or np.sum(labels>0)<20:
        continue

    all_tg_folder.append(sub_folder)
    files = []
    subject_id = []
    for fd in folders:
        filenames = os.listdir(datapath + '/' + fd)
        filenames = sorted(filenames)
        for f in filenames:
            subject_id.append(fd.split('_split')[0] + '_' + f.split('_run')[0])
            files.append(datapath + '/' + fd + '/' + f)

    ind = np.isin(subject_id, selectSub)
    og_subject_id = np.array(subject_id)[ind].tolist()
    og_files = np.array(files)[ind].tolist()

    # og_subject_id = subject_id
    # og_files = files

    print('tg num is %d, og num is %d' % (len(tg_files), len(og_files)))

    for m in range(3):

        # setting training model
        if m == 0:
            args.num_epochs = 20
            cross_val_model = False
            subject_id = og_subject_id
            files = og_files
            args.batch_size = 32
            args.lr = 1e-4
        else:
            args.num_epochs = 100
            cross_val_model = True
            subject_id = tg_subject_id
            files = tg_files
            args.batch_size = 16

        save_model = not cross_val_model
        if m == 1:
            finetune = False
            args.lr = 1e-4
        else:
            finetune = True
            args.lr = 1e-4

        SavePath = args.pretrain_dir
        mdfile = 'trained_model_exc_' + sub_folder[0].split('_split')[0] + '.pt'
        if not os.path.exists(SavePath):
            os.mkdir(SavePath)

        if 'adj' not in vars():
            # Get the graph structure
            X = 0
            for f in files:
                X += cross_val.Eloadmat(f, ['static_R'])[0]
            X = X / len(files)

            mean_connec = np.squeeze(cross_val.triu2mat(X))
            dist, idx = graph.distance_lshforest(mean_connec, k=10, metric='cosine')
            adj = graph.adjacency(dist, idx).astype(np.float32)
            adj = graph.rescale_adj(adj)

        labels = []
        for f in files:
            tmp = cross_val.Eloadmat(f, ['label'])
            labels.extend(tmp[0])
        labels = np.stack(labels, 0)

        # for batch nodes
        max_num_nodes = adj.shape[0]
        foldnum = 10
        all_vals = []
        all_val_pred = []
        all_val_subid = []
        all_val_labels = []

        seed = 777
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        uniq_subject_raw, uniq_index = np.unique(subject_id, return_index=True)
        uniq_lables_raw = labels[uniq_index]

        rind = np.random.permutation(len(uniq_subject_raw))
        uniq_subject = uniq_subject_raw.copy()[rind]
        uniq_labels = uniq_lables_raw.copy()[rind]

        if cross_val_model:
            for i in range(foldnum):

                train_dataset, val_dataset = cross_val.cross_train_test_loader(
                    files, subject_id, uniq_subject, args.batch_size, i, uniq_labels, foldnum)

                model = encoders.GCNSPnet(
                    input_dim, args.hidden_dim, args.output_dim, args.num_classes,
                    args.num_gc_layers, bn=args.bn, dropout=args.dropout, concat=False, args=args).cuda()

                if m==1:
                    traintype = 'baseline'
                else:
                    traintype = 'transfer'
                tgfolder = sub_folder[0].split('_split')[0]
                test_mdfile = 'trained_model_GCNSP_'+tgfolder+'_'+traintype+'_fold_' + str(i) + '.pt'

                if args.train_or_test=='train':
                    if finetune:
                        model.load_state_dict(torch.load(SavePath + '/' + mdfile))

                    _, val_accs, val_pred, val_subid, val_labels = train(train_dataset, adj, model, args,
                                                                         val_dataset=val_dataset, test_dataset=None,
                                                                         writer=writer)
                    torch.save(model.state_dict(), SavePath + '/' + test_mdfile)
                else:

                    model.load_state_dict(torch.load(SavePath + '/' + test_mdfile))

                    if 'data_adj' not in vars():
                        # csr matrix to torch.sparse
                        data_adj = adj
                        data_adj = data_adj.tocoo()
                        indices = np.vstack((data_adj.row, data_adj.col))
                        values = data_adj.data
                        shape = data_adj.shape

                        i = torch.LongTensor(indices)
                        v = torch.FloatTensor(values)
                        data_adj = Variable(torch.sparse.FloatTensor(i, v, torch.Size(shape)),
                                            requires_grad=False).cuda()

                    val_result = evaluate(val_dataset, data_adj, model, args, name='Validation')
                    val_accs = val_result['acc']
                    val_pred = val_result['pred_scores']
                    val_subid = val_result['subjectid']
                    val_labels = val_result['labels']

                all_vals.append(np.array(val_accs))
                all_val_pred.append(val_pred)
                all_val_subid.append(val_subid)
                all_val_labels.append(val_labels)

            all_vals = np.vstack(all_vals)

            if m == 1:
                print('For the site %d :' % ratio)
                if args.train_or_test == 'train':
                    print('with non-transfer GCNSP, final accuracy is %f, optimal accuracy is %f ' % ( all_vals.mean(axis=0)[-1], all_vals.mean(axis=0).max()))
                else:
                    print('with non-transfer GCNSP, final accuracy is %f ' % all_vals.mean(axis=0)[-1])
                all_baseline_acc.append(all_vals)
                all_baseline_pred.append(all_val_pred)
                all_baseline_subid.append(all_val_subid)
                all_baseline_labels.append(all_val_labels)
            else:

                if args.train_or_test == 'train':
                    print('with transfer GCNSP, final accuracy is %f, optimal accuracy is %f ' % (all_vals.mean(axis=0)[-1], all_vals.mean(axis=0).max()))
                else:
                    print('with transfer GCNSP, final accuracy is %f ' % all_vals.mean(axis=0)[-1])
                all_transfer_acc.append(all_vals)
                all_transfer_pred.append(all_val_pred)
                all_transfer_subid.append(all_val_subid)
                all_transfer_labels.append(all_val_labels)
            if writer is not None:
                writer.close()

            # all_results = {'all_transfer_acc': all_transfer_acc, 'all_baseline_acc': all_baseline_acc,
            #                'sample_ratio': sample_ratio, 'tf_pred': all_transfer_pred, 'tf_subid': all_transfer_subid,
            #                'tf_labels': all_transfer_labels, 'bl_pred': all_baseline_pred, 'bl_subid': all_baseline_subid,
            #                'bl_labels': all_baseline_labels,'tg_folder':all_tg_folder}
            #
            # scipy.io.savemat('all_results_leave_site_out_transfer_MDD_meta_and_ours_selectedSub_onlymeta_rmS4_epoch100_decreseLR20_40_allroi' + cudaflag+'_newcalc.mat', all_results)

        if save_model:
            if args.train_or_test=='test':
                continue
            train_dataset = cross_val.file2loader(files,args.batch_size)[0]
            val_dataset = cross_val.file2loader(files[:100],args.batch_size)[0]

            model = encoders.GCNSPnet(
                input_dim, args.hidden_dim, args.output_dim, args.num_classes,
                args.num_gc_layers, bn=args.bn, dropout=args.dropout, concat=False, args=args).cuda()

            _, val_accs_t, _, _, _ = train(train_dataset, adj, model, args, val_dataset=val_dataset)
            torch.save(model.state_dict(), SavePath + '/' + mdfile)

average_acc = 0
optimal_average_acc = 0
for tmp in all_baseline_acc:
    average_acc += tmp.mean(axis=0)[-1]/len(all_baseline_acc)
    optimal_average_acc += tmp.mean(axis=0).max() / len(all_baseline_acc)
average_acc2 = 0
optimal_average_acc2 = 0
for tmp in all_transfer_acc:
    average_acc2 += tmp.mean(axis=0)[-1]/len(all_transfer_acc)
    optimal_average_acc2 += tmp.mean(axis=0).max() / len(all_transfer_acc)

average_acc = []
optimal_average_acc = []
for tmp in all_baseline_acc:
    average_acc.append(tmp.mean(axis=0)[-1])
    optimal_average_acc.append(tmp.mean(axis=0).max())
average_acc2 = []
optimal_average_acc2 = []
for tmp in all_transfer_acc:
    average_acc2.append(tmp.mean(axis=0)[-1])
    optimal_average_acc2.append(tmp.mean(axis=0).max())

print('The final average accuracy for all sites:')
if args.train_or_test=='train':
    print('With non-transfer GCNSP, average accuracy is %f, optimal accuracy is %f' % (np.array(average_acc).mean(), np.array(optimal_average_acc).mean()))
    print('With transfer GCNSP, average accuracy is %f, optimal accuracy is %f' % (np.array(average_acc2).mean(), np.array(optimal_average_acc2).mean()))
else:
    print('With non-transfer GCNSP, average accuracy is %f' % (np.array(average_acc).mean()))
    print('With transfer GCNSP, average accuracy is %f' % (np.array(average_acc2).mean()))