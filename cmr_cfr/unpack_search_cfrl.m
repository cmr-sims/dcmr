function search = unpack_search_cfrl(res, experiment, fit)
%UNPACK_SEARCH_CFRL   Prepare results from a search for running a simulation.
%
%  search = unpack_search_cfrl(res, experiment)

search = [];
for i = 1:length(res)
    % get the best-fitting parameters from the final generation
    parameters = res(i).parameters;
    fitness = res(i).fitness;
    [fval, ind] = min(fitness);
    param_vec = parameters(ind,:);
    
    % unpack free and fixed parameters
    param = unpack_param(param_vec, res(i).fstruct.param_info);
    [~, fixed] = search_param_cfrl(res(i).fstruct.model_type, experiment);
    param = propval(fixed, param, 'strict', false);
    
    % set derived parameters
    simdef = sim_def_cfrl(experiment, fit);
    pool = load(simdef.pool_file);
    param = prep_param_cfrl(param, simdef, pool.category);
    param = check_param_cfrl(param);
    
    % save search information in standard format
    s = struct;
    s.experiment = experiment;
    s.fit = fit;
    s.model_type = res(i).fstruct.model_type;
    s.param = param;
    s.data = res(i).fstruct.data;
    s.fval = fval;
    s.parameters = parameters;
    s.fitness = fitness;
    s.names = {res(i).fstruct.param_info.name};
    s.param_info = res(i).fstruct.param_info;
    s.fixed = fixed;
    s.search_opt = res(i).search_opt;
    s.run_opt = res(i).run_opt;
    s.rseed = res(i).rseed;
    
    search = cat_structs(search, s);
end
