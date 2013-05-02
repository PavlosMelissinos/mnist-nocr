function [ x_labels ] = label_extend(labels, threshold)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

%lSize = size(labels);
[m,n] = size(labels);
x_labels = ones(m, 10);
x_labels = x_labels * (1-threshold)/9;
for i = 1:m
    x_labels(i,labels(i)+1) = threshold;
end
x_labels = [x_labels labels];
end
