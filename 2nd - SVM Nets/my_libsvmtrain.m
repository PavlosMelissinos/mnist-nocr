function [ SVMTrained, trainset_results, testset_results, time] = my_libsvmtrain( trainset, trainlabels, testset, testlabels, kerfuncmode, C, deg, gamma )
%UNTITLED Summary of this function goes here
% -s svm_type : set type of SVM (default 0)
% 	0 -- C-SVC
% 	1 -- nu-SVC
% 	2 -- one-class SVM
% 	3 -- epsilon-SVR
% 	4 -- nu-SVR
% -t kernel_type : set type of kernel function (default 2)
% 	0 -- linear: u'*v
% 	1 -- polynomial: (gamma*u'*v + coef0)^degree
% 	2 -- radial basis function: exp(-gamma*|u-v|^2)
% 	3 -- sigmoid: tanh(gamma*u'*v + coef0)
% -d degree : set degree in kernel function (default 3)
% -g gamma : set gamma in kernel function (default 1/num_features)
% -r coef0 : set coef0 in kernel function (default 0)
% -c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
% -n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
% -p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
% -m cachesize : set cache memory size in MB (default 100)
% -e epsilon : set tolerance of termination criterion (default 0.001)
% -h shrinking: whether to use the shrinking heuristics, 0 or 1 (default 1)
% -b probability_estimates: whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
% -wi weight: set the parameter C of class i to weight*C, for C-SVC (default 1)
% kerfuncmode:
% linear = '0';
% polynomial = '1';
% radial_basis = '2';
% sigmoid = '3';
    
tic;

kernel = num2str(kerfuncmode);
libsvm_options = strcat(['-t ' kernel ' -m 1000 -c ' num2str(C) ' -d ' num2str(deg) ' -g ' num2str(gamma)]);

%train SVM
SVMTrained = svmtrain(trainlabels, trainset, libsvm_options);
time = toc;

tic
%test SVM trainset
[predicted_label, accuracy, decision_values] = svmpredict(trainlabels, trainset, SVMTrained);
trainset_results = struct('predicted_label', predicted_label, 'accuracy', accuracy, 'decision_values', decision_values);
toc;
tic;
%test SVM testset
[predicted_label, accuracy, decision_values] = svmpredict(testlabels, testset, SVMTrained);
testset_results = struct('predicted_label', predicted_label, 'accuracy', accuracy, 'decision_values', decision_values);
toc;

end

