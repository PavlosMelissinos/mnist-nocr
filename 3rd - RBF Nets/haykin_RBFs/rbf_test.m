function [p, missed, mI] = rbf_test(w, x, c, sigma, d)

% function y = rbf_test(w, x, c, sigma)
%
%  Tests a regularized radial basis function network
%  
%    The input structure RBF must be of the format defined in rbf.m
%    Do a 'help rbf' to see the structure.
%
%    w - weights
%    x - test data
%    c - centers
%
% Hugh Pasika 1998

rx = size(x,1);

G = rbf_mkGF(x,c);
G = exp((-1/(2*sigma^2))*G.^2);
G = [G ones(rx,1)];

y = G*w;

[p,I] = rbf_correct(y, d);
missed = zeros(size(I,2),size(x,2));
for i = 1 : size(I,1)
    missedsample = I(i);
    missed(i,:) = x(missedsample,:);
end
mI = I;
