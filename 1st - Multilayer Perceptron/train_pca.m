function [ corTrain, corTest, ep_err, tr_time, W1, b1, W2, b2 ] = train_pca( TrainOrig, TestOrig, PC, TrainTarg, TestTarg, hN, oN, inD, lr, mom, epochs)
%train_pca returns the success rates for each train run, the ep_error
%   Detailed explanation goes here


% First we keep the first inD columns of the pc matrix, the ones with the
% maximum variance; The more columns we select, the more information will
% be kept and we will have better results. It's remarkably slower though.
SPC = PC( :, 1 : inD );

% then we multiply the original matrices with it to get the coordinates of
% the new points
SPCTrainset = TrainOrig * SPC; % first the trainset
SPCTestset  = TestOrig  * SPC; % and then the testset

% finally we append the targets to the result matrices, for
% training/testing purposes
xPCTrainset = [ SPCTrainset  TrainTarg ];
xPCTestset  = [ SPCTestset   TestTarg  ];

% initialize the weights and biases. Zero means random
W1 = 0;
b1 = 0;
W2 = 0;
b2 = 0;

% initialize some other stuff
start_epoch = 1;
%epochs_run  = 0;
trains_run  = 0;
epochs   = unique( epochs ); % remove any duplicates
corTrain = zeros( size( epochs ) ); % initialize: train set, test results
corTest  = zeros( size( epochs ) ); % initialize: test set, test results
ep_err   = zeros( size( epochs ) );
tr_time  = zeros( size( epochs ) );
total_tr_time = 0;
for i = 1:max(epochs)
    if ismember(i,epochs)
        end_epoch = i;
        step = end_epoch + 1 - start_epoch;
        tic;
        [ W1, b1, W2, b2, ep_erri ] = bpm_train( xPCTrainset, hN, oN, inD, lr, mom, step, W1, b1, W2, b2, 0 );
        tr_timei = toc;
        %istart = epochs_run + 1;
        %iend   = istart + step - 1;
        ep_err(start_epoch:end_epoch) = ep_erri;
        total_tr_time = total_tr_time + tr_timei;
        trains_run = trains_run + 1;
        tr_time( trains_run ) = total_tr_time;
        
        % run the tests for accuracy
        corTrain( trains_run ) = bpm_test( W1, b1, W2, b2, xPCTrainset );
        corTest( trains_run )  = bpm_test( W1, b1, W2, b2, xPCTestset );
        
        
        start_epoch = i + 1;
        %epochs_run = epochs_run + step;
    end
end
end