function [ corTrain, corTest, ep_err, tr_time ] = train_script( TrainOrig, TestOrig, PC, TrainTarg, TestTarg, hN, inD, epochs)
% function [ corTrain, corTest, ep_err ] =
% train_script( TrainOrig, TestOrig, PC, TrainTarg, TestTarg, hN, inD,
% epochs)
%

% hN = reshape(hN, 1, []);
m = size(hN, 2);
n = size(inD, 2);
p = size(unique(epochs), 2);
q = max(epochs);
corTrain = zeros(m, n, p);
corTest  = zeros(m, n, p);
ep_err   = zeros(m, n, q);
tr_time  = zeros(m, n, p);
for i = 1 : m
    hNi = hN(i);
    for j = 1 : n
        inDi = inD(j);
        [ corTraini, corTesti, ep_erri, tr_timei ] = train_pca( TrainOrig, TestOrig, PC, TrainTarg, TestTarg, hNi, 10, inDi, 0.1, 0.5, epochs);
        corTrain(i, j, :) = corTraini;
        corTest(i, j, :) = corTesti;
        ep_err(i, j, :) = ep_erri;
        tr_time(i, j, :) = tr_timei;
    end
end

