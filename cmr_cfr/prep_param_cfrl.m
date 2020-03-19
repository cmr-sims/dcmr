function param = prep_param_cfrl(param, simdef, category)
%PREP_PARAM_CFRL   Prepare parameters for a search or simulation.
%
%  This function sets special parameters that contain data, such as
%  semantic vectors. Other parameter fields determine whether and how
%  these data are actually used in the simulation; for example,
%  param.SC sets the weighting of localist category units.
%
%  param = prep_param_cfrl(param, simdef, category)

n_item = length(category);

% semantic cuing
if simdef.opt.qsem
    param.sem_mat = getfield(load(simdef.sem_mat_file, ...
                                  'sem_mat'), 'sem_mat');
else
    param.sem_mat = [];
end

% semantic context
if simdef.opt.sem
    param.sem_vec = getfield(load(simdef.sem_mat_file, ...
                                  'vectors'), 'vectors')';
else
    param.sem_vec = [];
end

% item localist
if simdef.opt.loc
    if ~exist('data', 'var')
        load(simdef.data_file)
    end
    param.loc_vec = eye(n_item);
else
    param.loc_vec = [];
end

% category localist
if simdef.opt.cat
    if ~exist('data', 'var')
        load(simdef.data_file)
    end
    ucat = unique(category);
    cat_vec = zeros(length(ucat), n_item);
    for i = 1:length(ucat)
        cat_vec(i,category==ucat(i)) = 1;
    end
    param.cat_vec = cat_vec;
else
    param.cat_vec = [];
end
