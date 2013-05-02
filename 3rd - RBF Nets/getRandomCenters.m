function [ centers, ss, list ] = getRandomCenters( k, trainset )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

%k = 10;
[ss, list] = sort( randn( 1, length( trainset ) ) );
list = list( 1 : k );
centers = trainset( list, : );
end
