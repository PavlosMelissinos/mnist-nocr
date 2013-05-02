function [ COEFF, SCORE, LATENT ] = train_pca( A )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

% normalize
% m = mean(A);
% AStd = A - repmat(m, size(A(:,1)));


%get the principal components
[ COEFF, SCORE, LATENT ] = princomp(AStd);

% PC1 = AStd * COEFF;

% PC = SCORE;



end

