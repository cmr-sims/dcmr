
%% fits using the MortPoly17 models

fits = {'base', 'wikiw2v_context' 'wikiw2v_item' 'wikiw2v_context_item'};

% test job
fits = {'base'};
experiments = {'cfr'};
flags = '-t 00:30:00 --mem=8gb --cpus-per-task=8 --partition=debug';
jobs = {};
for i = 1:2
    jobs{i} = submit_searches_cfrl(experiments, fits, flags, 'n_workers', 8);
end

% full search (based on JML paper search parameters)
fits = {'base' 'hybrid_' 'full_wikiw2v'};
flags = '-t 24:00:00 --mem=12gb --cpus-per-task=12';
n_rep = 15;
jobs = cell(1, n_rep);
for i = 1:n_rep
    jobs{i} = submit_searches_cfrl(experiments, fits, flags, 'n_workers', 12);
end

rep = load_rep_de_cfrl(jobs, true);
job = submit_indiv_best_params_cfrl(experiments, fits, '-t 04:00:00 --mem=4gb');
