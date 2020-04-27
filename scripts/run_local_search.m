


% what needs to be set up to run a basic search locally

res = indiv_search_cfrl('cfr','local_cat_wikiw2v',...
                        'proj_dir','~/Science/GitHub/cmr_cfr/', ...
                        'res_dir','~/Science/Analysis/catFR/sims/');

% modifications to indiv_search_cfrl (requirement that proj_dir and
% res_dir are specified)
% propagate this out to functions that call indiv_search_cfrl
% this includes submit_searches_cfrl, although that function just
% passes along the varargin

