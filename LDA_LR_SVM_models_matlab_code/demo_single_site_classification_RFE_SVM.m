clear all
% set data path
datapath = '../MDD_allroi';
folders = dir(fullfile(datapath,'MDD*'));
folders = {folders.name};

feat_ratio = 0.0001; % set feature ratio for feature selection
for s=1:length(folders)
    
    tppath=fullfile(datapath,folders{s});
    files=dir(fullfile(tppath,'*.mat'));
    if length(files)<10
        continue
    end
    % load data
    data=[];label_all=[];subject=[];
    for f=1:length(files)
        load(fullfile(tppath,files(f).name));
        data=[data;static_R];
        label_all(f,1)=label;
        subject{f,1}=subject_id;
    end
    
    %%% excluding subjects
    load('select_subject_final.mat')
    [la,lb]=ismember(subject,subject_ids);
    if sum(la)>=10
        data = data(la,:);
        label_all = label_all(la);
        subject = subject(la);
        [acc_mean{s},pred_scores{s},pred_labels{s}]=classification_SVM(data,label_all,subject,feat_ratio);
        all_subs{s} = subject;
        all_labels{s} = label_all;
    end
end

save('multisite_svm_ratio1_selectedSub_onlymeta.mat','acc_mean','pred_scores','pred_labels','all_subs','all_labels','folders');

for i=1:length(acc_mean)
    acc(i,1) = mean(mean(acc_mean{i},2));
end

disp('The average classification reults of single site via RFE-SVM :')
disp(['For RFE-SVM, accuracy is ' num2str(mean(acc))])
