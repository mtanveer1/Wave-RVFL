function [EVAL_Train,EVAL_Test,TrainTime,TestTime]  = Wave_RVFL_Model(trainX,trainY,testX,testY,option)
% function [model,train_accuracy,train_time] = ELM_train(trainX,trainY,option)
% TrainAcc=0;
%%%%%%%%%%%%%%%%% Training Starts %%%%%%%%%%%%%%%%%
N = option.N;
C = option.C;
s = 1;
activation = option.activation;

% beta1=option.beta1;     % exponential decay rates for the first moment estimate
% beta2=option.beta2;     % exponential decay rates for the second moment estimate
alpha=option.alpha;   % learning rate
% epsilon=option.epsilon; % small constant used to avoid division by zero
% max_iter=option.max_iter;  % maximum iteration number
m=option.m;          % mini batch size
a=option.a;           % a and b are loss parameter
b=option.b;
%%%%%%%%%%%%%%%%%
alltrain=[trainX,trainY];
l=size(alltrain,1);
% Set the random seed for reproducibility
seed = 0;
rng(seed);

% Get the number of rows in the matrix
numRows = size(alltrain, 1);

% Generate a random permutation of row indices
permIndices = randperm(numRows);

% Interchange rows of the matrix based on the permutation
randomizedMatrix = alltrain(permIndices, :);

rand_data=randomizedMatrix(1:m,:);

trainXrand=rand_data(:,1:end-1);
trainYrand=rand_data(:,end);

[Nsample,Nfea] = size(trainXrand);

tic

W = (rand(Nfea,N)*2*s-1);
bias = s*rand(1,N);
X1 = trainXrand*W+repmat(bias,Nsample,1);

if activation == 1
    X1 = sigmoid(X1);
elseif activation == 2
    X1 = sin(X1);
elseif activation == 3
    X1 = tribas(X1);
elseif activation == 4
    X1 = radbas(X1);
elseif activation == 5
    X1 = tansig(X1);
elseif activation == 6
    X1 = relu(X1);
end

X = [trainXrand,X1]; %Direct Link
% X=X1;
X = [X,ones(Nsample,1)];%bias in the output layer

%%%%%%%%%%%%%%%%%%%%%%%%%
% gamma=0.01*ones(m,1);  % initialize model parameter
r=0.01*ones(size(X,2),1);      % initialize first order moment
v=0.01*ones(size(X,2),1);      % initialize second order moment
beta = 0.01*zeros(size(X,2),1); %initilize beta
beta1=0.9;       % exponential decay rates for the first moment estimate
beta2=0.999;     % exponential decay rates for the second moment estimate
epsilon= 10^-8;  % small constant used to avoid division by zero
max_iter = 1000;  % maximum iteration number
tol = 10^-5;
%%%%%%%%%%%%%%%%%%%%%%%%%%

% PreviousLoss = inf;
betaPrevious = inf;

for t=1:max_iter

    
    Xi_Matrix=X*beta-trainYrand; %Xi matrix with respect to all samples
    temp3=zeros(size(X,2),1);

    for i=1:m
        temp1=exp(a*Xi_Matrix(i,:));
        temp_numerator=Xi_Matrix(i,:)*X(1,:)'*temp1*(2+a*Xi_Matrix(i,:));
        temp_denomenator=(1+b*(Xi_Matrix(i,:).^2)*temp1).^2;

        temp2=temp_numerator./temp_denomenator;
        temp3=temp3+temp2;
    end

    temp4=C*temp3;
    gradient=beta+temp4;

    % Update bias-corrected first and second moment estimates
    r = beta1 .* r + (1 - beta1) .* gradient;
    v = (beta2 .* v) + ((1 - beta2) .* (gradient.^2));
    r_hat = r ./ (1 - beta1^t);
    v_hat = v ./ (1 - beta2^t);
    beta = beta - ((alpha * r_hat) ./ (sqrt(v_hat) + epsilon));

    if norm(beta-betaPrevious)<tol
        fprintf('Converged at iteration %d\n', t);
        break
    else
        betaPrevious=beta;
    end

end

Predict_Y_train = sign(X*beta); %output of ELM

EVAL_Train = Evaluate(trainYrand,Predict_Y_train);

TrainTime=toc;


%%%%%%%%%%%%%%%%%%%% Testing Starts %%%%%%%%%%%%%%%%%%%%%

tic
% beta = model.beta;
% W = model.W;
% b = model.b;
% activation = 10;

Nsample = size(testX,1);


X1 = testX*W+repmat(bias,Nsample,1);


if activation == 1
    X1 = sigmoid(X1);
elseif activation == 2
    X1 = sin(X1);
elseif activation == 3
    X1 = tribas(X1);
elseif activation == 4
    X1 = radbas(X1);
elseif activation == 5
    X1 = tansig(X1);
elseif activation == 6
    X1 = relu(X1);
end


X = [testX,X1];
% X=X1;
X=[X,ones(Nsample,1)];

rawScore = X*beta;
f=sign(rawScore);

EVAL_Test = Evaluate(testY,f);
TestTime=toc;

end

%EOF
