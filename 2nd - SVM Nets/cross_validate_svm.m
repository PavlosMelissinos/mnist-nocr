function [ train_results, test_results, time ] = cross_validate_svm( trainset, testset, trainlabels, testlabels, C, gamma, degree )
%function [ train_results, test_results, time ] = cross_validate_svm( trainset, testset, trainlabels, testlabels, C, gamma, degree )
%	little script used to run cross validation for the given sets, with the given parameters
%   trainset, testset the original sets
%	trainlabels, testlabels, the original labels
%	the parameters can be in the form of vectors
%	C: cost parameters as it is used by svmtrain function
%	gamma: gamma parameter
%	degree: not used
%	3-nested for C, gamma, degree (in that order)
%	merges the train and test set and then splits them in 7 parts
%	which are then used for cross validation
%	train_results, test_results: 4d matrices [C, gamma, degree, cross-validation group] of structs
%	So, if input is C = [1 5 10 1000 1000000], gamma = [0.0005 0.001] (degree doesnt matter) 
%	then train_results(1,:,:,:) has all the results for C = 1
%	and train_results(1,2,:,:) has all the results for C = 1 and gamma = 0.001

xset = [trainset trainlabels; testset testlabels];
groups = cross_validation_split( xset );
groupsTotal  = size( groups, 1 );
testsamplesTotal = size( groups, 2 );
trainsamplesTotal = testsamplesTotal * ( groupsTotal - 1 );
featureTotal = size( groups, 3 );


trainsets = zeros( groupsTotal, trainsamplesTotal, featureTotal );
testsets  = zeros( groupsTotal, testsamplesTotal, featureTotal );
for i = 1 : size( groups, 1 )
    a = 1:size(groups, 1);
    a(i) = [];
    testsets(i, :, :) = reshape( groups(i,:,:), testsamplesTotal, featureTotal );
    trainsets(i, :, :) = reshape( groups(a,:,:), trainsamplesTotal, featureTotal );
end

% initialize the training results
%SVMTrained = zeros( size( C, 1 ), size( gamma, 1 ), size( degree, 1 ), size( groups, 1 ));
%train_results = zeros( size( C, 1 ), size( gamma, 1 ), size( degree, 1 ), size( groups, 1 ));
%test_results = zeros( size( C, 1 ), size( gamma, 1 ), size( degree, 1 ), size( groups, 1 ));
time = zeros( size( C, 1 ), size( gamma, 1 ), size( degree, 1 ), size( groups, 1 ));

for i = 1 : size( C, 2 );
    c = C(i);
    for j = 1 : size( gamma, 2 )
        g = gamma(j);
        for k = 1 : size( degree, 2 )
            d = degree(k);
            train_scores = zeros(size(groups,1),3);
            test_scores = zeros(size(groups,1),3);
            for l = 1 : size( groups, 1 )
                disp( strcat( [     'C = ' num2str(c)]));
                disp( strcat( [ 'gamma = ' num2str(g)]));
                disp( strcat( ['degree = ' num2str(d)]));
                disp( strcat( ['group = ' num2str(l)]));
                trainset = trainsets( l, :, 1 : featureTotal - 1 );
                trainset = reshape( trainset, trainsamplesTotal, featureTotal - 1 );
                trainlabels = trainsets( i, :, featureTotal );
                trainlabels = reshape( trainlabels, trainsamplesTotal, 1 );
                testset  = testsets( l, :, 1 : featureTotal - 1);
                testset = reshape(testset, testsamplesTotal, featureTotal - 1);
                testlabels  = testsets( l, :, featureTotal );
                testlabels = reshape( testlabels, testsamplesTotal, 1 );
                [ ~, train_resultsi, test_resultsi, timei] = my_libsvmtrain( trainset, trainlabels, testset, testlabels, num2str(2), c, d, g);
                %SVMTrained( i, j, k, l ) = SVMTrainedi;
                train_results( i, j, k, l ) = train_resultsi;
                test_results( i, j, k, l ) = test_resultsi;
                time( i, j, k, l ) = timei;
                train_scores(l,:) = train_resultsi.accuracy;
                test_scores(l,:) = test_resultsi.accuracy;
            end
            disp( strcat( [     'C = ' num2str(c) ':']));
            disp( strcat( [ 'gamma = ' num2str(g) ':']));
            disp( strcat( ['degree = ' num2str(d) ':']));
            tr_mean = mean(train_scores,1);
            tr_best = max(train_scores,[],1);
            te_mean = mean(test_scores,1);
            te_best = max(test_scores,[],1);
            disp( strcat( ['Average scores = ' num2str(tr_mean(1)) ' on training samples, ' num2str(te_mean(1)) ' on test samples.']));
            disp( strcat( ['Highest scores = ' num2str(tr_best(1)) ' on training samples, ' num2str(te_best(1)) ' on test samples.']));
        end
    end
end
%
end

