function [ corTrain, corTest, ep_err, tr_time ] = train_script2( TrainOrig, TestOrig, PC, TrainTarg, TestTarg, lr, mom, epochs)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

% hN = reshape(hN, 1, []);
m = size(lr, 2);
n = size(mom, 2);
p = size(unique(epochs), 2);
q = max(epochs);
corTrain = zeros(m, n, p);
corTest  = zeros(m, n, p);
ep_err   = zeros(m, n, q);
tr_time  = zeros(m, n, p);
for i = 1 : m
    lri = lr(i);
    for j = 1 : n
        momi = mom(j);
        [ corTraini, corTesti, ep_erri, tr_timei ] = train_pca( TrainOrig, TestOrig, PC, TrainTarg, TestTarg, 50, 10, 50, lri, momi, epochs);
        corTrain(i, j, :) = corTraini;
        corTest(i, j, :) = corTesti;
        ep_err(i, j, :) = ep_erri;
        tr_time(i, j, :) = tr_timei;
    end
end


end

