function [p, I] = rbf_correct(y,d)

% p - percent correct in two class problem
% y - output from testing phase
% d - desired output class (0 to 9)

perr = 0;
samples = size(y,1);
for i = 1:samples
    [~,Iy] = max(y(i,:));
    [~,Id] = max(d(i,:));
    if Iy ~= Id
        perr = perr + 1;
        I(perr) = i;
    end
end

p = 100 * (samples - perr) / samples;

fprintf(1,'The percent correct is:  %5.2f.\n', p)
