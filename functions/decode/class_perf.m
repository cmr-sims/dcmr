function perf = class_perf(category, evidence)
%CLASS_PERF   Calculate classifer performance from evidence.
%
%  perf = class_perf(category, evidence)

labels = category';
labels = labels(:);

ucat = unique(category);
n_cat = length(ucat);
targets = zeros(length(labels), n_cat);
for i = 1:n_cat
    targets(labels==ucat(i),i) = 1;
end

n_rep = size(evidence, 3);
perf = NaN(1, n_rep);
for i = 1:n_rep
    perfmet = perfmet_maxclass(evidence(:,:,i)', targets', struct());
    perf(i) = perfmet.perf;
end
