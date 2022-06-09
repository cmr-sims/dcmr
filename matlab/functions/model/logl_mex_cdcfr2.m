function logl = logl_mex_cdcfr2(param, data)
%LOGL_MEX_CDCFR2   Evaluate a model for multiple distraction conditions.
%
%  logl = logl_mex_cdcfr2(param, data)

if ~isfield(data, 'distract')
    data.distract = split_distract_cfrl(data);
end

% run separately for each distraction condition
logl = 0;
for i = 1:length(data.distract)
    d = data.distract{i};
    d_param = param_cdcfr2(param, d.distract_len);
    logl_distract = logl_mex_tcm(d_param, d);
    logl = logl + logl_distract;
end
