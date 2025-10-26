# Wave-RVFL
Wave-RVFL: A Randomized Neural Network Based on Wave Loss Function

Please cite the following paper if you are using this code.

Reference: 
M. Sajid, A. Quadir, and M. Tanveer, "Wave-RVFL: A Randomized Neural Network Based on Wave Loss Function." 
Published in the 31st International Conference on Neural Information Processing (ICONIP) 2024.
Arxiv Link: https://arxiv.org/abs/2408.02824
https://doi.org/10.1007/978-981-96-6579-2_17

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

The experimental hardware configuration includes a personal computer featuring an Intel(R) Xeon(R) Gold 6226R CPU with a clock speed of 2.90 GHz and 128 GB of RAM. The system runs on Windows 11 and utilizes Matlab2023a to run all the experiments.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

We have put a demo code of the "Wave-RVFL" model with the "adult" dataset.

For the demo purpose, following hyperparameters set used for the experiment. To get the optimal results, please tune the hyperparameters. For the detailed experimental setup, please refer to the supplementary material of the paper.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Demo Hyperparameter setting

option.C=1; %Regularization parameter

option.N=810; %Number of hidden nodes

option.activation = 1; %Sigmoid Activation function

option.a=-1;

option.b=0.6;

option.alpha=0.0001;

option.m=2^8;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Note: For detailed parameters setting, please refer "Wave-RVFL: A Randomized Neural Network Based on Wave Loss Function" paper.

Description of files:
Wave_RVFL_Main.m: This is the main file to run selected models on datasets.

Wave_RVFL_Model.m: This is a model function which is used to train and test.

sigmoid.m: Sigmoid activation function.

relu.m: ReLU activation function.

Selu.m: SeLU activation function.

Evaluate.m: Function to evaluate the accuracy.

adult.mat: adult dataset used to execute the code.

Remarks:

1. The codes have been cleaned for better readability and documented, then re-run and checked the codes only in a few datasets, so if you find any bugs/issues, please write to M. Sajid (phd2101241003@iiti.ac.in).

2. For the detailed experimental setup, please follow our paper.  
Some parts of the codes have been taken from:

1. M. Sajid, A. K. Malik, M. Tanveer and P. N. Suganthan, "Neuro-Fuzzy Random Vector Functional Link Neural Network for Classification and Regression Problems," in IEEE Transactions on Fuzzy Systems, vol. 32, no. 5, pp. 2738-2749, May 2024, doi: 10.1109/TFUZZ.2024.3359652. 

2. Mushir Akhtar, M. Tanveer, Mohd. Arshad, "Advancing Supervised Learning with the Wave Loss Function: A Robust and Smooth Approach," in  Pattern Recognition, Vol. 155, 110637, 2024, doi: 10.1016/j.patcog.2024.110637.
