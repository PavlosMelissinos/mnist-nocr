function [ groups ] = cross_validation_split( samples )
%UNTITLED Summary of this function goes here
% splits the set into 6 groups

groupTotal = 7;
sampleTotal  = size(samples, 1);
featureTotal = size(samples, 2);
groups = zeros(groupTotal,sampleTotal/groupTotal,featureTotal);
for i = 1:groupTotal
    groups(i,:,:) = samples(i:groupTotal:sampleTotal,:);
end

end

