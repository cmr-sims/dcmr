function simdef = sim_def_cfrl(experiment, fit, res_dir)
%SIM_DEF_CFRL   Get a simulation definition.
%
%  simdef = sim_def_cfrl(experiment, fit, res_dir)

if nargin < 3
    res_dir = '';
end

% unpack model information
info = get_fit_info_cfrl(fit, experiment, res_dir);
files = get_exp_info_cfrl(experiment, res_dir);
[param_info, fixed] = search_param_cfrl(info.model_type, experiment);
opt = model_features_cfrl(info.model_type);

% prepare information in standard format
simdef = struct();
simdef.model_type = info.model_type;
simdef.experiment = experiment;
simdef.fit = fit;
simdef.opt = opt;
simdef.fixed = fixed;
simdef.param_info = param_info;
simdef.param_name = {param_info.name};
simdef.data_file = files.data;
simdef.pool_file = files.pool;
if isempty(opt.sem_model)
    simdef.sem_mat_file = '';
else
    simdef.sem_mat_file = files.(opt.sem_model).mat;
end
