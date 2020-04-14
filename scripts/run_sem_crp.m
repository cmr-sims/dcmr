
% SMP testing out the semantic CRP code

% partner script is in cdcat_analysis/cdcatmr/beh_recall/scripts/run_sem_crp_cdcatmr.m

% addpath(genpath('~/Science/GitHub/EMBAM'))
% addpath('~/Science/GitHub/tcm/util');
% init_cfr

% sem-CRP on the behavioral data from CFR experiment

res_dir = '/Users/polyn/Science/GitHub/cfr/figs/';

% load the observed / real data
projpath = '/Users/polyn/Science/GitHub/cmr_cfr/';
datapath = '/data/cfr_eeg_mixed_data_clean.mat';
sempath = '/data/cfr_wikiw2v_raw.mat';
sembinpath = '/data/cfr_wikiw2v_bin.mat';

real = load(fullfile(projpath,datapath));
sem = load(fullfile(projpath,sempath));
bin = load(fullfile(projpath,sembinpath));

% this has already been run, this is the script that creates the 
% bin edges file (sembinpath file):
% create_sem_crp_bins(real.data, sem.sem_mat, sembinpath);

% this does preprocessing necessary to create the different
% versions of a sem_crp

[~, ind] = unique(real.data.pres_itemnos);
category = real.data.pres.category(ind);

[act, poss] = item_crp(real.data.recalls, ...
                       real.data.pres_itemnos, ...
                       real.data.subject, ...
                       length(sem.sem_mat));

print_sem_crp(act, poss, sem.sem_mat, bin.edges, bin.centers, ...
              fullfile(res_dir, 'cfr_sem_crp.eps'));
print_sem_crp(act, poss, sem.sem_mat, bin.edges, bin.centers, ...
              fullfile(res_dir, 'cfr_sem_crp_within.eps'), ...
              'mask', category == category');
print_sem_crp(act, poss, sem.sem_mat, bin.edges, bin.centers, ...
              fullfile(res_dir, 'cfr_sem_crp_between.eps'), ...
              'mask', category ~= category');