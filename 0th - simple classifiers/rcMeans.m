function [ xrcMeans, time ] = rcmeans( set, targets )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
tic;
rSet = reshape( set', 28, 28, size( set, 1 ) );

cMeans = mean( rSet, 1 );
cMeans = reshape( cMeans, 28, size( cMeans, 3 ));

rMeans = mean( rSet, 2 );
rMeans = reshape( rMeans, 28, size( rMeans, 3 ));

rcMeans = [ cMeans; rMeans ];

xrcMeans = [rcMeans' targets];
toc;
end

