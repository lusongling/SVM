function [eval_predicted]=svm(train_data,train_label,test_data)
[~,trn]=size(train_data);
[~,ten]=size(test_data);
Aeq=train_label';
beq=0;
lb=zeros(trn,1);
f=-ones(trn,1);
A=[];b=[];x0=[];
op=ones(trn);
p=10;
c=100;
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
w_data=((train_data(:,index)'*train_data+ones(m,trn)).^p)*(x.*train_label);
btr=train_label(index)-w_data;
b0=mean(btr);
%%
g=((test_data'*train_data+ones(ten,trn)).^p)*(x.*train_label)+repmat(b0,ten,1);
eval_predicted=sign(g);
ind0=find(eval_predicted==0);
eval_predicted(ind0)=1;
end