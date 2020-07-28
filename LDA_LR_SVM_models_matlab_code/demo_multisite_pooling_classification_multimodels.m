% set data path
datapath = '../MDD_allroi';
files = get_dirfiles(datapath,'*.mat');

% load data
data=[];label_all=[];subject=[];
for f=1:length(files)
    disp(num2str(f))
    load(files{f});
    data=[data;static_R];
    label_all(f,1)=label;
    subject{f,1}=subject_id;
end

%%% excluding subjects
load('select_subject_final.mat')
[la,lb]=ismember(subject,subject_ids);
sel_data = data(la,:);
sel_label = label_all(la);
sel_subject= subject(la);

feat_ratio = 0.0001; % set feature ratio for feature selection
[acc_mean1, pred_scores1, pred_labels1] = classification_LDA(sel_data,sel_label,sel_subject,feat_ratio);
[acc_mean2, pred_scores2, pred_labels2] = classification_LR(sel_data,sel_label,sel_subject,feat_ratio);
[acc_mean3, pred_scores3, pred_labels3] = classification_SVM(sel_data,sel_label,sel_subject,feat_ratio);

% calcualted AUC
for i=1:length(pred_labels1)
    pred1(i,:) = sum(pred_scores1{i});
    pred2(i,:) = sum(pred_scores2{i});
end
pred1 = sum(pred1);
pred2 = sum(pred2);
pred3 = mean(abs(pred_scores3).*pred_labels3);
label = sel_label;
label(label==-1)=0;
auc1 = AUC(label,pred1');
auc2 = AUC(label,pred2');
auc3 = AUC(label,pred3');

disp('multi-site pooling classification results:')
disp(['For RFE-LDA, accuracy is ' num2str(mean(mean(acc_mean1,2))) ' , AUC is ' num2str(auc1*100)])
disp(['For RFE-LR, accuracy is ' num2str(mean(mean(acc_mean2,2))) ' , AUC is ' num2str(auc2*100)])
disp(['For RFE-SVM, accuracy is ' num2str(mean(mean(acc_mean3,2))) ' , AUC is ' num2str(auc3*100)])
