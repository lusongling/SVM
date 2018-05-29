clear;
clc;
load('test.mat');
load('train.mat');
[ft,trn]=size(train_data);
[~,ten]=size(test_data);
Aeq=train_label';
beq=0;
lb=zeros(trn,1);
c=10^6;
ub=ones(trn,1)*c;
f=-ones(trn,1);
A=[];b=[];x0=[];
op=ones(trn);
i=0;
for p=[2,3,4,5]
    i=i+1;
Kp=(train_data'*train_data+op).^p;
H=(train_label*train_label').*Kp;
options =optimset('LargeScale','off','Maxiter',1000);
x=quadprog(H,f,A,b,Aeq,beq,lb,ub,x0,options);
%% choose the index of support vector
indx=x-ones(trn,1)*(1e-4);
indxdex=find(indx>=0);
indx_data=indx(indxdex);
[~,index_label]=min(indx_data);
index=indxdex(index_label);

%%
w_data=((train_data(:,index)'*train_data+ones(1,trn)).^p)*(x.*train_label);
b0=1/train_label(index)-w_data;
%%
train=(((train_data'*train_data+op).^p)*(x.*train_label)+repmat(b0,trn,1)).*train_label;
[acctrN,~]=size(find(train>=0));
acctr(i)=acctrN/trn;
test=(((test_data'*train_data+ones(ten,trn)).^p)*(x.*train_label)+repmat(b0,ten,1)).*test_label;
[accteN,~]=size(find(test>=0));
accte(i)=accteN/ten;
end

