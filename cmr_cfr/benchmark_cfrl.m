function [logl_mex, logl_mat, logl, logl_all] = benchmark_cfrl(ver)

if nargin < 1
    ver = 1;
end

% data for cfr, subject 1
data = getfield(load('cfr_benchmark_data.mat', 'data'), 'data');
sem = load('cfr_wikiw2v.mat');

% parameters taken from:
% ~/work/cfr/tcm/tcm_wikiw2v_qc/tcm_wikiw2v_qc_2018-06-28_stats.mat
param = struct();
param.B_enc = 0.79;
param.B_rec = 0.94;
param.P1 = 16.84;
param.P2 = 1.64;
param.G = 1;
param.X1 = 0.0093;
param.X2 = 0.32;
param.C = 1.22;
param.D = 100;
param.T = 4.64;
param.B_s = 0.1;
param.S = 2.5;
param.I = 0;
param.stop_rule = 'op';
param.init_item = 0;
param.B_ipi = 0;
param.B_ri = 0;
param.Dfc = 70.46;
param.Dcf = 100;
param.Afc = 0;
param.Acf = 1.22;
param.Sfc = 0;
param.Lfc = 1;
param.Lcf = 1;
param.Scf = 2.5;

disp('Testing implementations with parameters from wikiw2v_context fit...')

disp('No semantics:')
logl_mex = logl_mex_tcm(param, data);
if ver == 1
    [logl, logl_all] = logl_tcm(param, data);
else
    param.pre_vec = eye(768);
    [logl, logl_all] = logl_tcm(param, data);
end
logl_mat = nansum(logl(:));
fprintf('  logl mex: %.8f\n', logl_mex)
fprintf('  logl mat: %.8f\n', logl_mat)
if abs(logl_mex - logl_mat) < .001
    disp('  Differences are less than 0.001.')
else
    disp('  Warning: Differences are greater than 0.001.')
end

disp('With semantics:')
param.sem_mat = sem.sem_mat;
logl_mex = logl_mex_tcm(param, data);
if ver == 1
    logl = logl_tcm(param, data);
else
    param.pre_vec = eye(768);
    logl = logl_tcm(param, data);
end
logl_mat = nansum(logl(:));
fprintf('  logl mex: %.8f\n', logl_mex)
fprintf('  logl mat: %.8f\n', logl_mat)
if abs(logl_mex - logl_mat) < .001
    disp('  Differences are less than 0.001.')
else
    disp('  Warning: Differences are greater than 0.001.')
end


% try generating some data
%seq = gen_tcm(param, data);
