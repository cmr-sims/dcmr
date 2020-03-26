function param = param_cdcfr2(param, distract_len)
%PARAM_CDCFR2   Set parameters that are specific to distraction length.
%
%  param = param_cdcfr2(param, distract_len)

if distract_len == 2.5
    dname = 'd1';
elseif distract_len == 7.5
    dname = 'd2';
else
    return
end

for i = 1:length(param.distract_params)
    f = param.distract_params{i};
    param.(f) = param.([f '_' dname]);
end
