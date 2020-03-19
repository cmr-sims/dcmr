function split = split_distract_cfrl(data)
%SPLIT_DISTRACT_CFRL   Split a data struct into distraction conditions.
%
%  split = split_distract_cfrl(data)

distract = data.pres.distractor(:,1);
udistract = unique(distract);

split = cell(1, length(udistract));
for i = 1:length(udistract)
    trials = distract==udistract(i);
    d = trial_subset(trials, data);
    d.recalls_vec = recalls_vec_tcm(d.recalls, data.listLength);
    d.distract_len = udistract(i);
    d.trials = trials;
    split{i} = d;
end
