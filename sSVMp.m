clear;
clc;
load('test.mat');
load('train.mat');
[ft,trn]=size(train_data);
[~,ten]=size(test_data);
Aeq=train_label';
beq=0;
lb=zeros(trn,1);
f=-ones(trn,1);
A=[];b=[];x0=[];
op=ones(trn);
for p=[1,2,3,4,5]
    i=0;
    for c=[0.1,0.6,1.1,2.1]
            i=i+1;
Kp=(train_data'*train_data+op).^p;
H=(train_label*train_label').*Kp;
ub=ones(trn,1)*c;
options =optimset('LargeScale','off','Maxiter',1000);
x=quadprog(H,f,A,b,Aeq,beq,lb,ub,x0,options);
%% choose the index of support vector
index1=find(x<=c);
index2=find(x>1e-4);
index=intersect(index1,index2);
[m,~]=size(index);
%%
w_data=((train_data(:,index)'*train_data+ones(m,2000)).^p)*(x.*train_label);
btr=train_label(index)-w_data;
b0=mean(btr);
%%
train=(((train_data'*train_data+op).^p)*(x.*train_label)+repmat(b0,trn,1)).*train_label;
[acctrN,~]=size(find(train>=0));
acctr(p,i)=acctrN/trn;
test=(((test_data'*train_data+ones(ten,trn)).^p)*(x.*train_label)+repmat(b0,ten,1)).*test_label;
[accteN,~]=size(find(test>=0));
accte(p,i)=accteN/ten;
    end
end