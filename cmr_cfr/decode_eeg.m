function [evidence, perf] = decode_eeg(pat)
%DECODE_EEG   Decode stimulus category based on patterns of oscillatory power.
%
%  Use a ridge regression classifier to decode category from measures
%  of oscillatory power, using cross-validation at the level of lists.
%
%  [evidence, perf] = decode_eeg(pat)
%
%  INPUTS
%  pat - pattern struct
%      Aperture-format pattern with EEG data, saved using
%      export_pattern. Events must include fields for session,
%      trial, and category.
%
%  OUTPUTS
%  evidence - [trials x categories] numeric array
%      Classifier evidence for each category. Categories are in the
%      sorting order of the category codes.
%
%  perf - [1 x lists] numeric array
%      Classifier performance by list.

pattern = flatten_pattern(pat.mat);
events = pat.dim.ev.mat;
[list, values] = make_index([events.session], [events.trial]);

targets = zeros(length(events), 3);
labels = [events.category];
ucat = unique(labels);
for i = 1:length(ucat)
    targets(labels==ucat(i),i) = 1;
end

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
