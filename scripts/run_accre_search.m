function run_accre_search(type)
%RUN_ACCRE_SEARCH
%
% type = 'basic_fits';
% type = 'hybrid_fit';


switch type
  case 'basic_fits'
    % combinations of different components of context
    fits = {'local' 'cat' 'wikiw2v' ...
            'local_cat' 'local_wikiw2v' 'cat_wikiw2v' ...
            'local_cat_wikiw2v'};
    experiments = {'cfr'};
    flags = '-t 06:00:00 --mem=12gb --cpus-per-task=12';
    jobs = {};
    n_rep = 10;
    for i = 1:n_rep
      jobs{end+1} = submit_searches_cfrl(experiments, fits, flags, ...
                                         'n_workers', 12, ...
                                         'proj_dir', '~/github/cmr_cfr', ...
                                         'res_dir', '~/work');
    end

  case 'hybrid_fit'
    
    % custom version with semantic cuing AND distributed context
    fits = {'hybrid_dc_loc_cat_qi'};
    experiments = {'cfr'};
    flags = '-t 06:00:00 --mem=12gb --cpus-per-task=12';
    jobs = {};
    n_rep = 10;
    for i = 1:n_rep
      jobs{end+1} = submit_searches_cfrl(experiments, fits, flags, ...
                                         'n_workers', 12, ...
                                         'proj_dir', '~/github/cmr_cfr', ...
                                         'res_dir', '~/work');
    end
end