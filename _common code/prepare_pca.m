function [ newTrainSet, newTestSet ] = prepare_pca( trainset, testset, inD )
%UNTITLED3 Summary of this function goes here
% inD is how many principal components to make

%get the principal components
[ COEFF, ~, LATENT ] = princomp(trainset);

sumLatent = cumsum(LATENT)./sum(LATENT);

disp(strcat(['Keeping ' num2str(100*sumLatent(inD)) '% of the original information.']));
% First we keep the first inD columns of the pc matrix, the ones with the
% maximum variance; The more columns we select, the more information will
% be kept and we will have better results. It's remarkably slower though.
SPC = COEFF( :, 1 : inD );

% then we multiply the original matrices with it to get the coordinates of
% the new points
newTrainSet = trainset * SPC; % first the trainset
newTestSet  = testset  * SPC; % and then the testset

end

