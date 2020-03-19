function param = check_param_cfrl(param)
%CHECK_PARAM_CFRL   Prepare parameters for running TCM in CFRL.
%
%  param = check_param_cfrl(param)

% get non-numeric settings from the model type string
opt = model_features_cfrl(param.model_type);

if ~isfield(param, 'stop_rule')
    param.stop_rule = opt.stop_rule;
end

if ~isfield(param, 'B_s')
    param.B_s = 0;
end

if ~isfield(param, 'I')
    param.I = 0;
end

if ~isfield(param, 'init_item')
    param.init_item = opt.init_item;
end

if ~isfield(param, 'B_ri')
    if isfield(param, 'B_ipi')
        param.B_ri = param.B_ipi;
    else
        param.B_ri = 0;
    end
end

% translate the G parameter (a ratio) to Dfc (a value for
% pre-experimental context relative to a fixed learning rate of 1)
if isfield(param, 'G') && ~isempty(param.G) && ~isnan(param.G)
    param.Dfc = (1 - param.G) / param.G;
    param.G = NaN;
end

if isfield(param, 'D')
    param.Dcf = param.D;
end

if isfield(param, 'C')
    param.Acf = param.C;
end

if isfield(param, 'S')
    param.Scf = param.S;
end

if ~isempty(param.loc_vec) || ~isempty(param.cat_vec) || ~isempty(param.sem_vec)
    x = [param.SL * param.loc_vec
         param.SC * param.cat_vec
         param.SD * param.sem_vec];
    param.pre_vec = x ./ sqrt(sum(x.^2, 1));
else
    param.pre_vec = [];
end

param = check_param_tcm(param);
