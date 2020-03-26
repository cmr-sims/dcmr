function stats = model_comp_cfrl(fits, experiment)
%MODEL_COMP_CFRL   Compare models of data from an experiment.
%
%  stats = model_comp_cfrl(fits, experiment)

% get the number of subjects
files = get_exp_info_cfrl(experiment);
s = load(files.data);
n_subj = length(unique(s.data.subject));

n_model = length(fits);
V = NaN(1, n_model);
logl = NaN(n_subj, n_model);
for i = 1:n_model
    % load latest fits for this model
    info = get_fit_info_cfrl(fits{i}, experiment);
    s = load(info.res_file);
    
    % get the number of datapoints for each subject
    if i == 1
        n = NaN(length(s.stats), 1);
        for j = 1:n_subj
            d = s.stats(j).data;
            n(j) = nnz(d.recalls) + size(d.recalls, 1);
        end
    end
    
    % get the number of free parameters for this model
    V(i) = length(s.stats(1).param_info);
    
    % fit for this subject
    for j = 1:n_subj
        logl(j,i) = -s.stats(j).fitness;
    end
end

% AIC statistic by subject
[saic, swaic] = aic_subj(logl, n, V);

% output all statistics
stats = struct;
stats.experiment = experiment;
stats.fits = fits;
stats.n = n;
stats.V = V;
stats.logl = logl;
stats.saic = saic;
stats.swaic = swaic;
