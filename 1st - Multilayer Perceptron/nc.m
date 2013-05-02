function [ rate, time ] = nc( test, train, testlabels, trainlabels, step )


tic;
dtrain = double( train ) / 255;
dtest = double( test ) / 255;
dtest = dtest( 1 : step : end , : );
dtrain = dtrain( 1 : step : end , : );
testlabels = testlabels( 1 : step : end , : );
trainlabels = trainlabels( 1 : step : end , : );
% total number of test samples
ttltestsmpls = size( dtest, 1 );
hits = 0;

% we create a matrix of size 10x784 to store the centroids
clusters = zeros(10, size(dtrain,2));
% and a vector to store the total number of each cluster's members
totals = zeros(10, 1);
% this loop gives us the centroid for each cluster
for i = 1 : size(dtrain, 1)
    currentClass = trainlabels(i);
    currentClassIndex = currentClass + 1;
    % first we compute the old sum again
    sum = clusters(currentClassIndex, :) * totals(currentClassIndex);
    % then we add the new value to it
    sum = sum + dtrain(i,:);
    % and increment the total number of samples for this cluster by one
    totals( currentClassIndex ) = totals( currentClassIndex ) + 1;
    % finally, divide the sum with the new total number of samples
    clusters( currentClassIndex, : ) = sum / totals( currentClassIndex );
end
% this loop computes the distance of each test sample with the centroid of
% the clusters
for i = 1 : size(dtest, 1)
    
    dists = zeros(size(clusters,1),1);
    currentClass = testlabels(i,1);
    % for each row of the centroids matrix
    for j = 1 : size(clusters,1)
        
        % get the euclidean distance between the two row vectors
    	dists(j,1) = norm(clusters(j,:) - dtest(i,:));

    end
    % we get the index of the minimum element of the distance vector, which
    % incidentally is the selectedClass ( a number between 0 and 9 )
    [C, selectedClassIndex] = min(dists);
    selectedClass = selectedClassIndex - 1;
    if selectedClass == currentClass
        hits = hits + 1;
    end
end
rate = hits * 100 / ttltestsmpls;
time = toc;
end