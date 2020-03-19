function seq = gen_cdcfr2(param, data, n_rep)
%GEN_CDCFR2   Simulate data for multiple distraction conditions.
%
%  seq = gen_cdcfr2(param, data, n_rep)

[n_trials, n_items, n_recalls] = size_frdata(data);
seq = zeros(n_trials * n_rep, n_items);

if ~isfield(data, 'distract')
    data.distract = split_distract_cfrl(data);
end

% generate sequences for all distraction conditions
for i = 1:length(data.distract)
    % get data for this subset of trials
    d = data.distract{i};
    d_param = param_cdcfr2(param, d.distract_len);
    d_seq = gen_tcm(d_param, d, n_rep);
    seq(repmat(d.trials, [n_rep 1]), 1:size(d_seq, 2)) = d_seq;
end

% remove padding
cols = ~all(seq==0, 1);
seq = seq(:,cols);
