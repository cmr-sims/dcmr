function info = get_fit_info_cfrl(fit, experiment, res_dir)
%GET_FIT_INFO_CFRL   Get information about CFRL model fits.
%
%  info = get_fit_info_cfrl(fit, experiment, res_dir)

if nargin < 3
    res_dir = '';
end

files = get_exp_info_cfrl(experiment, res_dir);

if strcmp(fit, 'data')
    info.res_dir = files.res_dir;
    info.stat_file = files.data;
    return
end

switch fit
  case 'base'
    model_type = 'tcm';
  case 'wikiw2v_context'
    model_type = 'tcm_wikiw2v_qc';
  case 'wikiw2v_item'
    model_type = 'tcm_wikiw2v_qi';
  case 'wikiw2v_context_item'
    model_type = 'tcm_wikiw2v_qci';
  case 'dc_wikiw2v'
    model_type = 'tcm_dc_wikiw2v';
  case 'dc_ncf_wikiw2v'
    model_type = 'tcm_dc_ncf_wikiw2v';
  case 'dc_wikiw2v_item'
    model_type = 'tcm_dc_wikiw2v_qi';
  case 'hybrid_wikiw2v_item'
    model_type = 'tcm_dc_loc_wikiw2v_qi';
  case 'hybrid_wikiw2v'
    model_type = 'tcm_dc_loc_wikiw2v';
  case 'full_wikiw2v'
    model_type = 'tcm_dc_loc_cat_wikiw2v';
  case 'local'
    model_type = 'tcm_dc_loc';
  case 'cat'
    model_type = 'tcm_dc_cat';
  case 'wikiw2v'
    model_type = 'tcm_dc_wikiw2v';
  case 'local_cat'
    model_type = 'tcm_dc_loc_cat';
  case 'local_wikiw2v'
    model_type = 'tcm_dc_loc_wikiw2v';
  case 'cat_wikiw2v'
    model_type = 'tcm_dc_cat_wikiw2v';
  case 'local_cat_wikiw2v'
    model_type = 'tcm_dc_loc_cat_wikiw2v';
  case {'local_cat_wikiw2v_dsl' ...
        'local_cat_wikiw2v_dsc' ...
        'local_cat_wikiw2v_dsd'}
    c = regexp(fit, '_', 'split');
    model_type = ['tcm_dc_loc_cat_wikiw2v_' c{end}];
    
  otherwise
    error('Unknown fit type: %s', fit);
end

info.model_dir = files.model_dir;
info.model_type = model_type;
if isempty(files.model_dir)
    info.res_dir = '';
    return
end
info.res_dir = fullfile(files.model_dir, model_type);

% attempt to find the latest search results
try
    info.raw_file = get_search_file_cfrl(files.model_dir, model_type);
    [pathstr, name, ext] = fileparts(info.raw_file);
    
    % if there is a merged file, use that for the default res file
    info.res_name = name;
    info.res_file = fullfile(info.res_dir, [name '_merge.mat']);
    if ~exist(info.res_file, 'file')
        info.res_file = info.raw_file;
    end
    info.stat_file = fullfile(info.res_dir, [name '_stats.mat']);
catch
    info.res_file = '';
    info.res_name = '';
    info.stat_file = '';
end
