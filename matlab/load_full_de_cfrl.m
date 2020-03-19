function res = load_full_de_cfrl(job, experiments, fits)
%LOAD_FULL_DE_CFRL   Load a set of searches.
%
%  res = load_full_de_cfrl(job, experiments, fits)

if ischar(experiments)
    experiments = {experiments};
end
if ischar(fits)
    fits = {fits};
end

try
    out = fetchOutputs(job);
catch
    fprintf('There was a problem loading job %d\n', job.ID);
    res = [];
    return
end

if length(out) ~= length(fits)
    error('Fits must be the same length as job.Tasks')
end

res = struct;
for i = 1:length(out)
    info = get_fit_info_cfrl(fits{i}, experiments{i});
    f = [experiments{i} '_' fits{i}];

    if ~isfield(out{i}, 'fitness')
        res.(f) = [];
        fprintf('Search did not finish for %s.\n', f);
        continue
    end
    
    stats = unpack_search_cfrl(out{i}, experiments{i}, fits{i});
    
    res.(f).fitness = cat(1, stats.fitness);
    res.(f).parameters = cat(1, stats.parameters);
    res.(f).names = stats(1).names;
    res.(f).stats = stats;
end
