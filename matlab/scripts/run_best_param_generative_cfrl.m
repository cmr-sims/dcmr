
% code to take aggregate results files and split them apart by
% model variant

% Setting the path (specific to SMP computer, change this for your env)
cd ~/Science/GitHub/EMBAM
init_embam
cd ~/Science/GitHub/tcm
init_tcm
cd ~/Science/GitHub/cmr_cfr
init_cfr

% load('~/Science/Analysis/catFR/sims/res_local_cat_wikiw2v_variants.mat')

% which has fields: 
% res.cfr_local
% res.cfr_local_cat
% res.cfr_local_cat_wikiw2v

% want a file in the same directory called 'cfr_local' and when you
% load it you get the variables that are attached to that cfr_local
% structure just as separate variables.  did this, now have:
% 'cfr_local.mat', 'cfr_local_cat.mat', 'cfr_local_cat_wikiw2v.mat'
% corresponding fit codes are:
% 'local', 'local_cat', and 'local_cat_wikiw2v'

% script to take simulation optimization results structures and run
% generative simulations

cd ~/Science/Analysis/catFR

expt = 'cfr';
fit = 'local';
proj_dir = '~/Science/GitHub/cmr_cfr/';
res_dir = '~/Science/Analysis/catFR/sims/';
res_file = 'cfr_local.mat';

data = run_res_indiv_best_params_cfrl(expt, fit, proj_dir, ...
                                      res_dir, res_file, 'n_rep', 10);

fit = 'local_cat';
res_file = 'cfr_local_cat.mat';

data = run_res_indiv_best_params_cfrl(expt, fit, proj_dir, ...
                                      res_dir, res_file, 'n_rep', 10);

fit = 'local_cat_wikiw2v';
res_file = 'cfr_local_cat_wikiw2v.mat';

data = run_res_indiv_best_params_cfrl(expt, fit, proj_dir, ...
                                      res_dir, res_file, 'n_rep', 10);

% once these are all run, have files:
% 'cfr_local_stats.mat'
% 'cfr_local_cat_stats.mat'
% 'cfr_local_cat_wikiw2v_stats.mat'
% now can make summary statistics