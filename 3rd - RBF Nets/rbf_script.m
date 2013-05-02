function [train_results, test_results, time ]= rbf_script( trainset, train_targets, testset, test_targets, sigma, lambda, k, mode )
%UNTITLED Summary of this function goes here
%   if mode = 1 then random centers
%   if mode = 2 then k-means
%   else fuzzy c-means
time = zeros(size(sigma,2), size(k,2));
train_results = zeros(size(sigma,2), size(k,2));
test_results = zeros(size(sigma,2), size(k,2));
for si = 1:size(sigma,2)
    for ki = 1:size(k,2)
        fprintf(1,'sigma = %5.5f,\t', sigma(si));
        fprintf(1,'k = %5.5f\n', k(ki));
        if mode == 1
            train_resultsI = zeros(3,1);
            test_resultsI = zeros(3,1);
            for i = 1:3
                centers = getRandomCenters( k(ki), trainset );
                [ w, t ] = rbfTrainScript( trainset, train_targets, centers, sigma(si), lambda);
                train_resultsI(i) = rbf_test(w, trainset, centers, sigma(si), train_targets);
                test_resultsI(i) = rbf_test(w, testset, centers, sigma(si), test_targets);
                time(si,ki) = t;
            end
            train_resultsi = mean(train_resultsI);
            test_resultsi = mean(test_resultsI);
        else
            tic;
            if mode == 2
                [~, centers] = kmeans(trainset,k);
            else % mode == 0
                centers = fcm(trainset, k(ki));
            end
            fprintf(1,'Centers found in %d s.\t',toc);
            [ w, t ] = rbfTrainScript( trainset, train_targets, centers, sigma(si), lambda);
            timei = toc;
            fprintf(1,'Trained in %d s.\n',toc);
            train_resultsi = rbf_test(w, trainset, centers, sigma(si), train_targets);
            test_resultsi = rbf_test(w, testset, centers, sigma(si), test_targets);
            time(si,ki) = timei;
        end
        train_results(si, ki) = train_resultsi;
        test_results(si, ki) = test_resultsi;
    end
end
    

end

