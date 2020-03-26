function init_cfr()
%INIT_CFR   Add cmr_cfr functions to the path.
%
%  init_cfr()

proj_dir = fileparts(mfilename('fullpath'));

addpath(fullfile(proj_dir, 'functions', 'analysis'))
addpath(fullfile(proj_dir, 'functions', 'decode'))
addpath(fullfile(proj_dir, 'functions', 'model'))
addpath(fullfile(proj_dir, 'scripts'))

