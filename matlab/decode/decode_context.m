function [evidence, perf] = decode_context(context, category)
%DECODE_CONTEXT   Use pattern classification to decode stimulus category.
%
%  Use a ridge regression classifier to decode category from states
%  of context, using cross-validation at the level of lists.
%
%  evidence = decode_context(context, category)
%
%  INPUTS
%  context - [lists x items] cell array
%      The state of context predicted to be active during each trial.
%
%  category - [lists x items] numeric array
%      The category code for each trial.
%
%  OUTPUTS
%  evidence - [trials x categories] numeric array
%      Classifier evidence for each category. Categories are in the
%      sorting order of the category codes.

% set up decoding
ucat = unique(category);
labels = category';
labels = labels(:);
targets = zeros(length(labels), 3);
for i = 1:length(ucat)
    targets(labels==ucat(i),i) = 1;
end
[n_list, n_trial] = size(category);
list = repmat([1:n_list]', [1 n_trial])';
list = list(:);

c = context';
pattern = cat(2, c{:})';

% run cross-validation
opt = struct;
opt.f_train = @train_logreg;
opt.train_args = {struct('penalty', 10)};
opt.f_test = @test_logreg;
opt.verbose = false;
opt.runpar = true;
res = xval(pattern, list, targets, opt);

% unpack results
evidence = NaN(size(targets, 1), size(targets, 2));
for i = 1:length(res.iterations)
    test_ind = res.iterations(i).test_idx;
    evidence(test_ind,:) = res.iterations(i).acts';
end

perf = mean_evidence(res);
