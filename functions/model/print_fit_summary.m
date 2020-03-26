function print_fit_summary(fit, experiment, res_dir)
%PRINT_FIT_SUMMARY   Print plots for a number of analyses.
%
%  print_fit_summary(fit, experiment, res_dir)

files = get_exp_info_cfrl(experiment);
sem = load(files.wikiw2v.raw);
info = get_fit_info_cfrl(fit, experiment);
close all
if nargin < 3
    if isfield(info, 'res_file')
        res_dir = info.stat_file(1:end-4);
    else
        res_dir = fullfile(info.res_dir, 'data');
    end
end

load(info.stat_file);
if ~exist(res_dir, 'dir')
    mkdir(res_dir)
end

% standard catfr analyses
print_spc_cat(data, fullfile(res_dir, 'spc_cat.eps'), 'spc');
print_spc_cat(data, fullfile(res_dir, 'pfr_cat.eps'), 'pfr');
print_crp(data, fullfile(res_dir, 'crp.eps'));
print_crp_cat(data, 1, 'cat', fullfile(res_dir, 'crp_within_cat.eps'));
print_crp_cat(data, 2, 'cat', fullfile(res_dir, 'crp_from_cat.eps'));
print_crp_cat(data, 3, 'cat', fullfile(res_dir, 'crp_to_cat.eps'));

% category clustering
rec_mask = make_clean_recalls_mask2d(data.recalls);
s_lbc = lbc(data.pres.category, data.rec.category, data.subject, ...
            'recall_mask', rec_mask);

% probability of within-category transition, conditional on the
% category of the previous recall
print_clust_cat(data, fullfile(res_dir, 'clust_cat.eps'));

% semantic crp
bin = load(files.wikiw2v.bin);
pool = load(files.pool);
[act, poss] = item_crp(data.recalls, ...
                       data.pres_itemnos, ...
                       data.subject, ...
                       length(sem.sem_mat));
print_sem_crp(act, poss, sem.sem_mat, bin.edges, bin.centers, ...
              fullfile(res_dir, 'sem_crp.eps'));
print_sem_crp(act, poss, sem.sem_mat, bin.edges, bin.centers, ...
              fullfile(res_dir, 'sem_crp_within.eps'), ...
              'mask', pool.category == pool.category');
print_sem_crp(act, poss, sem.sem_mat, bin.edges, bin.centers, ...
              fullfile(res_dir, 'sem_crp_between.eps'), ...
              'mask', pool.category ~= pool.category');

if isfield(data.pres, 'distractor') && ...
        length(unique(data.pres.distractor)) > 1
    print_sem_crp_distract(data, pool, sem, bin, ...
                           fullfile(res_dir, 'sem_crp_distract.eps'));
    
    print_spc_distract(data, fullfile(res_dir, 'spc_distract.eps'), 'spc');
    print_spc_distract(data, fullfile(res_dir, 'pfr_distract.eps'), 'pfr');
    print_crp_distract(data, fullfile(res_dir, 'crp_distract.eps'));
    print_clust_distract(data, fullfile(res_dir, 'lbc_distract.eps'), 'lbc');
    print_clust_distract(data, fullfile(res_dir, 'tf_distract.eps'), 'tempfact');
end
