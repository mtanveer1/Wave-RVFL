%%
% Please cite the following paper if you are using this code.
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Reference %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% M. Sajid, A. Quadir, and M. Tanveer. "Wave-RVFL: A Randomized Neural Network Based on Wave Loss Function." 
% Published in the 31st International Conference on Neural Information Processing (ICONIP) 2024.
% Arxiv Link: https://arxiv.org/abs/2408.02824
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The experimental hardware configuration includes a personal computer featuring
% an Intel(R) Xeon(R) Gold 6226R CPU with a clock speed of 2.90 GHz and
% 128 GB of RAM. The system runs on Windows 11 and utilizes Matlab2023a
% to run all the experiments.
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% We have put a demo code of the "Wave-RVFL" model with the "adult" dataset 
% 
% For the demo purpose, following hyperparameters set used for the experiment. 
% To get the optimal results, please tune the hyperparameters.
% For the detailed experimental setup, please refer to the supplementary material of the paper.
%
% C=0.01;
% Number of Hidden Node=123;
% Act=3;
% Wave_a=4.5;
% Wave_b=1.6;
% Alpha=1;
% m=2^5;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
clc;
clear;
warning off all;
format compact;

%% Data Preparation
split_ratio=0.6; nFolds=5; 
% addpath(genpath('C:\Users\HP\OneDrive - IIT Indore\Desktop\Wave-RVFL\Models\RVFL\Code\GitHub'))
temp_data1=load('adult.mat');

temp_data=temp_data1.adult;

[Cls,~,~] = unique(temp_data(:,end));
No_of_class = size(Cls,1);


trainX=temp_data(:,1:end-1); mean_X = mean(trainX,1); std_X = std(trainX);
trainX = bsxfun(@rdivide,trainX-repmat(mean_X,size(trainX,1),1),std_X);
All_Data=[trainX,temp_data(:,end)];

[samples,~]=size(All_Data);
rng('default')
test_start=floor(split_ratio*samples);
training_Data = All_Data(1:test_start-1,:); testing_Data = All_Data(test_start:end,:);
test_x=testing_Data(:,1:end-1); test_y=testing_Data(:,end);
train_x=training_Data(:,1:end-1); train_y=training_Data(:,end);


% option.C=0.0001;
% option.N=3;
% option.activation=3;
% option.a=-2;
% option.b=0.6;
% option.alpha=0.0001;
% option.m=length(training_Data);

option.C=0.0001;
option.N=203;
option.activation=5;
option.a=-1;
option.b=0.6;
option.alpha=0.0001;
option.m=2^8;


[EVAL_Train,EVAL_Test,train_time,valid_time] = Wave_RVFL_Model(train_x,train_y,test_x,test_y,option);


% fprintf(1, 'Testing Accuracy of Wave-RVFL model is: %f\n', EVAL_Test(1,1));



