function varargout = load_rep_de_cfrl(jobs, save_results)
%LOAD_REP_DE_CFRL   Load a replicated search.
%
%  Looks over a number of search replications, and gets the
%  best-fitting parameters for each individual subject. The
%  experiments and fits variables must match the tasks within each
%  job, so that the results are stored correctly.
%
%  res = load_rep_de_cfrl(jobs, save_results)

if nargin < 2
    save_results = false;
end

if ~iscell(jobs)
    jobs = num2cell(jobs);
end

in = fetchInputs(jobs{1});
experiments = in(:,1);
for i = 1:length(experiments)
    experiments{i} = strrep(experiments{i}, '-', '_');
end
fits = in(:,2);

fprintf('Loading replications...');
r = cell(1, length(jobs));
for i = 1:length(jobs)
    fprintf('%d ', i);
    r{i} = load_full_de_cfrl(jobs{i}, experiments, fits);
end
fprintf('\n');

% collapse results from all successful jobs
rc = cat(1, r{:});

% for each model and subject, find the best fitness
timestamp = datestr(now, 30);
f = fieldnames(rc);
n_model = length(f);
n_subj = length(rc(1).(f{1}).stats);
model_logl = NaN(n_subj, n_model);
for i = 1:length(f)
    fprintf('Loading results for %s fit of %s...\n', ...
            fits{i}, experiments{i});
    info = get_fit_info_cfrl(fits{i}, experiments{i});

    % get summary stats across the replications
    rmc = cat(1, rc.(f{i}));
    if isempty(rmc)
        fprintf('%s: no searches finished.\n', f{i});
        continue
    end
    
    fitness = cat(2, rmc.fitness);
    parameters = cat(3, rmc.parameters);
    rep_stats = cat(1, rmc.stats);
    
    n_subj = size(fitness, 1);
    stats = [];
    for j = 1:n_subj
        % find which replication gave the best fit for this subject
        [~, rep] = min(fitness(j,:));
        
        % get the stats corresponding to that fit
        subj_stats = rmc(rep).stats(j);
        subj_stats.rep = rep;
        model_logl(j,i) = -subj_stats.fitness;
        stats = cat_structs(stats, subj_stats);
    end

    n_rep = size(rep_stats, 1);
    rseed = NaN(n_subj, n_rep);
    for j = 1:n_subj
        for k = 1:n_rep
            rseed(j,k) = rep_stats(k,j).rseed.Seed;
        end
    end
    
    fprintf('%s: %d finished.\n', f{i}, size(fitness, 2));

    % save out best fit information, aggregated over all replications
    if ~exist(info.res_dir, 'dir')
        mkdir(info.res_dir);
    end
    model_type = stats(1).model_type;
    filename = sprintf('%s_%s.mat', model_type, timestamp);
    res_file = get_next_file(fullfile(info.res_dir, filename));
    if save_results
        save(res_file, 'stats');
    end

    if nargout > 0
        res.(f{i}).fitness = fitness;
        res.(f{i}).parameters = parameters;
        res.(f{i}).stats = stats;
        res.(f{i}).rep_stats = rep_stats;
        res.(f{i}).rep_seed = rseed;
        res.(f{i}).task_seed = rseed(1,:);
    end
end

res.logl = model_logl;
if save_results
    file = sprintf('search_rep_%s.mat', timestamp);
    rep_file = get_next_file(fullfile(info.model_dir, file));
    if nargout > 0
        save(rep_file, 'res', '-v7.3');
    end
end

if nargout > 0
    varargout{1} = res;
else
    varargout = {};
end


function out_file = get_next_file(file)
%GET_NEXT_FILE   Add serial numbers until a new file is obtained.
%
%  out_file = get_next_file(file)

[pathstr, name, ext] = fileparts(file);
out_name = name;
n = 0;
while exist(fullfile(pathstr, [out_name ext]), 'file')
  n = n + 1;
  out_name = sprintf('%s%d', name, n);
end
out_file = fullfile(pathstr, [out_name ext]);
