function [ w, time ] = rbfTrainScript( trainset, traintargets, centers, sigma, lambda )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

%sigma = 4;
%lambda = 1;
tic;
w = rbf( trainset, centers, traintargets, sigma, lambda );
time = toc;
end

