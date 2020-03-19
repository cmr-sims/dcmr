function job = submit_decode_cfrl(experiment, sim_experiment, fit, ...
                                  res_name, w, n_workers, overwrite, ...
                                  subj_ind, flags)
%SUBMIT_DECODE_CFRL   Submit jobs to run decoding analysis in parallel.
%
%  job = submit_decode_cfrl(experiment, fit, res_name, w,
%      n_workers, overwrite, subj_ind, flags)

info = get_fit_info_cfrl(fit, experiment);
[par, base, ext] = fileparts(info.res_file);

cluster = parallel.cluster.Generic();
cluster.JobStorageLocation = '~/runs';
cluster.IntegrationScriptsLocation = '~/matlab/accre';
cluster.AdditionalProperties.AdditionalSubmitArgs = [' ' flags];

opt = struct;
opt.sim_experiment = sim_experiment;
opt.overwrite = overwrite;
opt.n_workers = n_workers;
opt.sigmas = 0:.025:.2;
opt.n_rep_optim = 10;
opt.n_rep_final = 100;
opt.plot_optim = false;

job = createJob(cluster);
for i = subj_ind
    outfile = fullfile(par, sprintf('%s_%s_%d.mat', base, res_name, i));
    opt.subj_ind = i;
    task = createTask(job, @decode_cfrl, 1, ...
                      {experiment, fit, res_name, w, opt});
end

submit(job);
