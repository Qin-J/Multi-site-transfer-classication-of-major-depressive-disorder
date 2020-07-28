function [acc_fold, label_predict_all,label_predict_all1] = classification_SVM(data,label,subject,feat_ratio)
% data is the orignial feature matrix, with n x m dimension, where n
% represent samples and m represent feautures 
% label is the n x 1 vector (-1 or 1)
% subject is the subject ID for each sample
addpath('libsvm_matlab')
fold =10;
repeat = 5;

% Calculate the number of positive and negative samples
[nSmp nFea]=size(data);
[uniq_subject, IA, IC]=unique(subject);
nSmp = length(uniq_subject);
indPos=find(label(IA)==1);
indNeg=find(label(IA)==-1);


nPos=length(indPos);
nNeg=length(indNeg);
nFoldPos=floor(nPos/fold);
nFoldNeg=floor(nNeg/fold);

label_predict_all=zeros(repeat,nSmp);
label_predict_all1=zeros(repeat,nSmp);
rand('seed',777); % Set random seed
for r=1:repeat
    randSmp=randperm(nSmp);

     % k-fold cross-validation 
    for f=1:fold
        disp(['repeat = ' num2str(r) '/' num2str(repeat) ' , fold = ' num2str(f) '/' num2str(fold)])
        
        if f==fold
            indFoldPos = [f*nFoldPos-nFoldPos+1:nPos];
            indFoldNeg = [f*nFoldNeg-nFoldNeg+1:nNeg];
        else
            indFoldPos = [f*nFoldPos-nFoldPos+1:f*nFoldPos];
            indFoldNeg = [f*nFoldNeg-nFoldNeg+1:f*nFoldNeg];
        end
        indTest = randSmp([indPos(indFoldPos);indNeg(indFoldNeg)]);
        
        % for multi runs of the same subject
        [la lb]=ismember(subject,uniq_subject(indTest));
        indTest = find(la);

        % Divide training and test sets according to the k-fold index 
        label_train = label;
        label_train(indTest,:)=[];
        label_test = label(indTest,:);
        data_train = data;
        data_train(indTest,:)=[];
        
        % feature selection based on ttest2 on training set
        [h p] = ttest2(data_train(find(label_train==1),:),data_train(find(label_train==-1),:));
        [p lbp]= sort(p);
        pnum = floor(size(data_train,2)*feat_ratio);
        data_train = data_train(:,lbp(1:pnum));
        data_test = data(indTest,lbp(1:pnum));

        % training classifer and then predicting 
        model=svmtrain(label_train,double(data_train),'-s 0 -t 0 -c 1 -b 0');
        [label_predict accurate probility] = svmpredict(label_test, double(data_test), model, '-b 0');
        acc_fold(r,f)=accurate(1);
        label_predict_all(r,indTest)=probility(:,1);
        label_predict_all1(r,indTest)=label_predict;
    end
end


   
     