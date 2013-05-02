function [rate, time] = nn(test, train, testlabels, trainlabels, k, step)
%UNTITLED Summary of this function goes here
%   test is a vector 1 x n
%   train is a 2-dimensional matrix m x n
%   k is the number of nearest neighbours to be computed (not working yet)

% total number of test samples
tic;
dtrain = double(train)/255;
dtest = double(test)/255;
dtest = dtest( 1 : step : end , : );
dtrain = dtrain( 1 : step : end , : );
testlabels = testlabels( 1 : step : end , : );
trainlabels = trainlabels( 1 : step : end , : );
ttltestsmpls = size(dtest,1);
% this is the variable that stores the success rate of the knn algorithm
hits = 0;
misses = 0;

for i = 1 : size(dtest,1)
    % first make a vector of zeros, this will be used to store the
    % euclidean distance between the test vector and each row of the train
    % matrix
    dists = zeros(size(dtrain,1),1);
    currentClass = testlabels(i,1);
    
    % then we compute the distance between a test vector and a train matrix
    
    % for each row of the train matrix
    for j = 1 : size(dtrain,1)

        % get the euclidean distance between the two vectors (rows)
    	dists(j,1) = norm(dtrain(j,:) - dtest(i,:));

    end
    % so now the 'dists' table holds the distances between our vector and the
    % training samples
    dists = [dists trainlabels];
    % we sort the distances ascendingly
    [Y,I] = sort(dists(:,1),1,'ascend');
    dists = dists(I,:);
    % then we get the first k rows
    kNNs = dists(1:k,:);
    % then we get the most frequent element of the second column
    selectedClass = mode(kNNs(:,2));
    % if the class we think is right is indeed the correct one
    if currentClass == selectedClass
        % we increment the true hits
        hits = hits + 1;
    else
        misses = misses + 1;
    end;
end
rate = hits * 100 / ttltestsmpls;
time = toc;
end