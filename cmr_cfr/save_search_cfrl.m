function [res_file, stats] = save_search_cfrl(res, experiment, fit)
%SAVE_SEARCH_CFRL   Save individual search results to disk.
%
%  [res_file, stats] = save_search_cfrl(res, experiment, fit)

% get search output file
model_type = res(1).fstruct.model_type;
timestamp = datestr(now, 30);
filename = sprintf('%s_%s.mat', model_type, timestamp);
info = get_fit_info_cfrl(fit, experiment);
if ~exist(info.res_dir, 'dir')
    mkdir(info.res_dir);
end
res_file = get_next_file(fullfile(info.res_dir, filename));

% unpack search results into full format
stats = unpack_search_cfrl(res, experiment, fit);
save(res_file, 'stats', 'res');
