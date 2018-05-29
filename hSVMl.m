clear;
clc;
load('test.mat');
load('train.mat');
[ft,trn]=size(train_data);
[~,ten]=size(test_data);
K=train_data'*train_data;
Aeq=train_label';
beq=0;
lb=zeros(trn,1);
c=10^6;
ub=ones(trn,1)*c;
f=-ones(trn,1);
H=(train_label*train_label').*K;
A=[];b=[];x0=[];
options =optimset('LargeScale','off','Maxiter',1000);
x=quadprog(H,f,A,b,Aeq,beq,lb,ub,x0,options);
%% choose the index of support vector
indx=x-ones(trn,1)*(1e-4);
indxdex=find(indx>=0);
indx_data=indx(indxdex);
[~,index_label]=min(indx_data);
index=indxdex(index_label);
%%
w0=train_data*(x.*train_label);
b0=1/train_label(index)-w0'*train_data(:,index);
%% accurancy of test and train data
test=(w0'*test_data+repmat(b0,1,ten))'.*test_label;
[accteN,~]=size(find(test>=0));
accte=accteN/ten;
train=(w0'*train_data+repmat(b0,1,trn))'.*train_label;
[acctrN,~]=size(find(train>=0));
acctr=acctrN/trn;