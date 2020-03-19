function job = submit_searches_cfrl(experiments, fits, flags, varargin)
%SUBMIT_SEARCHES_CFRL   Submit a job to run multiple searches.
%
%  Runs individual subject fits.
%
%  job = submit_searches_cfrl(experiments, fits, flags, ...)
%
%  INPUTS:
%  experiments - cell array of strings
%      Names of experiments to run.
%
%  fits - cell array of strings
%      Names of model fits to run.
%
%  flags - char
%      Flags to set options for sbatch.
%
%  OUTPUTS:
%  job - parallel.job.CJSIndependentJob
%      Handle to the submitted job.
%
%  OPTIONS:
%  f_logl - function_handle - @tcm_general_mex
%      Handle to function to evaluate likelihood, of the form:
%          logl = f_logl(param, data)
%
%  f_check_param - function_handle - @check_param_cfrl
%      Handle to function that completes a parameter struct by
%      setting defaults, etc., of the form:
%          param_out = f_check_param(param_in)
%
%  n_search - int - 1
%      Number of times to replicate each search, with different
%      random starting points.
%
%  n_workers - int - 1
%      Number of parallel workers to use when optimizing parameters
%      for diferrent subjects.

% set up a cluster
cluster = parallel.cluster.Generic();
cluster.JobStorageLocation = '~/runs';
cluster.IntegrationScriptsLocation = '~/matlab/accre';
cluster.AdditionalProperties.AdditionalSubmitArgs = [' ' flags];

job = createJob(cluster);
job.Name = mfilename;

for i = 1:length(experiments)
    for j = 1:length(fits)
        task = createTask(job, @indiv_search_cfrl, 1, ...
                          {experiments{i}, fits{j}, varargin{:}});
        set(task, 'CaptureDiary', true);
    end
end

submit(job);
fprintf('Job submitted.\n');
