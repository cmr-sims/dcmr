addpath ~/matlab/pmvpa/core/learn
addpath ~/matlab/pmvpa/core/util

demo_dir = '/Users/morton/Dropbox/work/cmr_cfr/cfr/figs/v1/demo'
%%
patterns = load(fullfile(demo_dir, 'patterns.txt'));
imagesc(patterns)
%%
df = readtable(fullfile(demo_dir, "labels.csv"));
labels = df.target;
chunks = df.chunk;
train = chunks == 1;
test = chunks == 2;
%%
targets = zeros(size(labels, 1), 3);
for i = 1:3
    targets(:, i) = labels == i;
end
%%
scratchpad = train_logreg(patterns(train, :)', targets(train, :)', struct('penalty', 10));
%%
[prob, scratchpad] = test_logreg(patterns(test, :)', targets(test, :)', scratchpad);
%%
out_dir = fullfile(demo_dir, 'matlab');
if ~exist(out_dir, 'dir')
    mkdir(out_dir)
end
probability = prob';
save(fullfile(out_dir, 'probability_logreg.txt'), 'probability', '-ascii');