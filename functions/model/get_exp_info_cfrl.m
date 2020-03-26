function files = get_exp_info_cfrl(experiment, res_dir, check)
%GET_EXP_INFO_CFRL   Get files needed to run a simulation of an experiment.
%
%  files = get_exp_info_cfrl(experiment, res_dir)
%
%  INPUTS
%  res_dir - char
%      Base results directory. Results for individual experiments
%      will be saved here.

if nargin < 3
    check = false;
end
if nargin < 2
    res_dir = '';
end

% assume all data is stored within the code project
proj_dir = fileparts(mfilename('fullpath'));

sem_exp = experiment;
if isempty(res_dir)
    files.res_dir = '';
    files.model_dir = '';
else
    files.res_dir = fullfile(res_dir, experiment);
    files.model_dir = fullfile(res_dir, experiment, 'tcm');
end
files.data_dir = fullfile(proj_dir, 'data');

switch experiment
  case 'cfr'
    files.data = fullfile(files.data_dir, 'cfr_eeg_mixed_data_clean.mat');
    files.data_raw = fullfile(files.data_dir, 'cfr_eeg_mixed_data.mat');
    files.pool = fullfile(files.data_dir, 'cfr_pool.mat');
  case 'cdcfr2'
    % full dataset
    files.data = fullfile(files.data_dir, 'cdcfr2_data_clean.mat');
    files.data_raw = fullfile(files.data_dir, 'cdcfr2_data.mat');
    files.pool = fullfile(files.data_dir, 'cdcfr2_pool.mat');
  case {'cdcfr2_d0' 'cdcfr2_d1' 'cdcfr2_d2'}
    % one distractor condition at a time
    c = regexp(experiment, '_', 'split');
    distract = c{2};
    datafile = sprintf('cdcfr2_data_clean_%s.mat', distract);
    files.data = fullfile(files.data_dir, datafile);
    files.data_raw = fullfile(files.data_dir, 'cdcfr2_data.mat');
    files.pool = fullfile(files.data_dir, 'cdcfr2_pool.mat');
    sem_exp = 'cdcfr2';
  case 'cdcfr2-1'
    files.data = fullfile(files.data_dir, 'cdcfr2_data_clean_s1.mat');
    files.data_raw = fullfile(files.data_dir, 'cdcfr2_data.mat');
    files.pool = fullfile(files.data_dir, 'cdcfr2_pool.mat');
    sem_exp = 'cdcfr2';
  case 'cdcfr2-2'
    files.data = fullfile(files.data_dir, 'cdcfr2_data_clean_s2.mat');
    files.data_raw = fullfile(files.data_dir, 'cdcfr2_data.mat');
    files.pool = fullfile(files.data_dir, 'cdcfr2_pool.mat');
    sem_exp = 'cdcfr2';
  otherwise
    error('Unknown experiment: %s.', experiment)
end

sem_models = {'wikiw2v'};
names = {'wikiw2v'};
for i = 1:length(sem_models)
    s.name = names{i};
    s.mat = fullfile(files.data_dir, ...
                     sprintf('%s_%s.mat', sem_exp, sem_models{i}));
    s.raw = fullfile(files.data_dir, ...
                     sprintf('%s_%s_raw.mat', sem_exp, sem_models{i}));
    s.bin = fullfile(files.data_dir, ...
                     sprintf('%s_%s_bin.mat', sem_exp, sem_models{i}));
    files.(sem_models{i}) = s;
end

if check
    check_paths(files);
end


function check_paths(files)

f = fieldnames(files);
for i = 1:length(f)
    if strcmp(f{i}, 'name')
        continue
    end
    
    if isstruct(files.(f{i}))
        check_paths(files.(f{i}))
    elseif ~exist(files.(f{i}), 'file') && ~exist(files.(f{i}), 'dir')
        fprintf('Warning: path does not exist: %s\n', files.(f{i}));
    end
end