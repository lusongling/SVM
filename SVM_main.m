clear;
clc;
load('train.mat');
load('eval.mat');
[eval_predicted]=svm(train_data,train_label,eval_data);
