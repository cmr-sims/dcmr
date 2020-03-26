function create_sem_crp_bins(data, sem_mat, edge_file, varargin)
%CREATE_SEM_CRP_BINS   Create bins for semantic CRP analyses.
%
%  create_sem_crp_bins(data, sem_mat, edge_file, ...)

def.n_per_subj = 10;
def.min_spacing = 0.05;
opt = propval(varargin, def);

% get count of transitions between individual items and count of
% times that each transition could have happened, conditional on
% prior recalls and remaining available items
[actual, possible] = item_crp(data.recalls, data.pres_itemnos, ...
                              data.subject, length(sem_mat));
min_n = length(unique(data.subject)) * opt.n_per_subj;

% get similarity values that were possible at least once in the
% experiment
x = -sem_mat(any(possible, 3));

% make bins with a minimum spacing constraint and a minimum bin count
start = min(x):opt.min_spacing:max(x);
[edges, centers] = make_dist_bins_adapt(x, start, min_n);
edges = sort(-edges);
centers = sort(-centers);
n = histc(-x, edges);

% save to disk for use with sem crp analyses
save(edge_file, 'edges', 'centers', 'n');
