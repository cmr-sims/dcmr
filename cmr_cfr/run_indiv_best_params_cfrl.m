function data = run_indiv_best_params_cfrl(experiment, fit, varargin)
%RUN_INDIV_BEST_PARAMS_CFR   Run simulations with best-fitting parameters.
%
%  Load the best-fitting parameters from a search and run the
%  generative version of the model to create synthetic data.
%
%  data = run_indiv_best_params_cfrl(experiment, fit, ...)
%
%  OPTIONS:
%  n_rep - int - 100
%      Number of times to replicate the lists for each subject.

def.n_rep = 100;
[opt, custom] = propval(varargin, def);
custom = propval(custom, struct, 'strict', false);

% load fit information
info = get_fit_info_cfrl(fit, experiment);
fprintf('Loading best parameters from: %s\n', info.res_file);
stats = getfield(load(info.res_file, 'stats'), 'stats');

% load data
simdef = sim_def_cfrl(experiment, fit);
data = getfield(load(simdef.data_file, 'data'), 'data');
pool = load(simdef.pool_file);
custom = prep_param_cfrl(custom, simdef, pool.category);

subject = unique(data.subject);
n_subj = length(subject);
net_data = [];
for i = 1:n_subj
    fprintf('%d ', i);
    
    % get full parameters, apply any customizations
    param = stats(i).param;
    param = propval(custom, param, 'strict', false);
    param = check_param_cfrl(param);
    
    % prep subject data for simulation
    subj_data = trial_subset(data.subject == subject(i), ...
                             rmfieldifexist(data, 'recalls_vec'));
    
    % run simulation
    if isfield(subj_data.pres, 'distractor') && ...
            length(unique(subj_data.pres.distractor)) > 1
        seq = gen_cdcfr2(param, subj_data, opt.n_rep);
    else
        seq = gen_tcm(param, subj_data, opt.n_rep);
    end

    % prepare full data structure for analysis
    subj_net_data = subj_data;
    if opt.n_rep > 1
        subj_net_data = repmat_struct(subj_net_data, opt.n_rep);
    end
    subj_net_data.recalls = seq;

    % update recall fields that can be derived from study data (e.g. itemno)
    subj_net_data = update_derived_fields(subj_net_data);

    % add to the full data structure
    net_data = cat_data(net_data, subj_net_data);
end
fprintf('\n');

% save results
data = net_data;
data.listLength = data.listLength(1);
fprintf('Saving statistics in: %s\n', info.stat_file);
save(info.stat_file, 'data', 'custom', 'simdef');


function s_full = repmat_struct(s, n_rep)

    s_full = s;
    f = fieldnames(s);
    for i = 1:length(f)
        if isstruct(s.(f{i}))
            s_full.(f{i}) = repmat_struct(s.(f{i}), n_rep);
        elseif ~isscalar(s.(f{i}))
            s_full.(f{i}) = repmat(s.(f{i}), n_rep, 1);
        end
    end

function data = update_derived_fields(data)

    if isfield(data, 'rec_itemnos') && isfield(data, 'pres_itemnos')
        data.rec_itemnos = study_mat2recall_mat(data.pres_itemnos, ...
                                                data.recalls, 0, NaN);
    end
    if isfield(data, 'rec_items') && isfield(data, 'pres_items')
        data.rec_items = study_mat2recall_mat(data.pres_items, ...
                                              data.recalls, {}, {});
    end
    if isfield(data, 'rec') && isfield(data, 'pres')
        if isfield(data.rec, 'category') && isfield(data.pres, 'category')
            data.rec.category = study_mat2recall_mat(data.pres.category, ...
                                                     data.recalls, 0, NaN);
        end
        
        if isfield(data.rec, 'listtype') && isfield(data.pres, 'listtype')
            data.rec.listtype = study_mat2recall_mat(data.pres.listtype, ...
                                                     data.recalls, 0, NaN);
        end
    end
