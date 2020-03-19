function [logl, logl_all, net] = logl_cdcfr2(param, data)
%LOGL_CDCFR2   Evaluate a model for multiple distraction conditions.
%
%  [logl, logl_all, net] = logl_cdcfr2(param, data)

[n_trials, n_items, n_recalls] = size_frdata(data);
logl = NaN(n_trials, n_recalls + 1);
logl_all = NaN(n_trials, n_recalls + 1, n_items + 1);

if ~isfield(data, 'distract')
    data.distract = split_distract_cfrl(data);
end
net = cell(1, length(data.distract));

for i = 1:length(data.distract)
    % simulate this distraction condition
    d = data.distract{i};
    d_param = param_cdcfr2(param, d.distract_len);
    [d_logl, d_logl_all, d_net] = logl_tcm(d_param, d);
    logl(d.trials,:) = d_logl;
    logl_all(d.trials,:,:) = d_logl_all;
    net{i} = d_net;
end
