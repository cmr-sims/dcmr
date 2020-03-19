function outfile = decode_cfrl(experiment, fit, res_name, w, varargin)
%DECODE_CFRL   Decode EEG and context.
%
%  outfile = decode_cfrl(experiment, fit, res_name, w, ...)
%
%  INPUTS
%  experiment : char
%      Name of the experiment to process.
%
%  fit : char
%      Name of the model fit to use.
%
%  res_name : char
%      Name of the results. Used in creating the output file.
%
%  w : double
%      Weighting of noise on item features. Item noise will be 
%      w * sigma, while context noise will be sigma.
%
%  OPTIONS
%  subj_ind : numeric array : []
%      Index of the subject to include, within the sorted list of subject
%      numbers. If specified, will create an output file specific to that
%      subject. Otherwise, all subjects will be saved in one file.
%
%  overwrite : boolean : false
%      If true, will overwrite existing output files.
%
%  n_workers : int : []
%      If not empty, will start a parallel pool with this number of
%      workers. If not specified, all cores on the system will be used.
%
%  sim_experiment : cell array of strings : {}
%      Experiment codes to use for running simulations. This can be used to
%      run different conditions with different parameters. Simulated data
%      will be merged before classification, so that it is run across all
%      conditions.
%
%  Other options will be passed to decode_context_match.

def = struct();
def.subj_ind = [];
def.overwrite = false;
def.n_workers = [];
def.sim_experiment = {};
[opt, optim_opt] = propval(varargin, def);

if ~isempty(opt.n_workers) && opt.n_workers > 1
    [pool, cluster] = job_parpool(opt.n_workers);
end

% experiment info
info = get_exp_info_cfrl(experiment);
load(info.data);

% output file
if isempty(opt.sim_experiment)
    % base on the results file that we load fitted parameters from
    fit_info = get_fit_info_cfrl(fit, experiment);
    [par, base, ext] = fileparts(fit_info.res_file);
else
    % base on the first results file; if all were loaded at the same time,
    % they'll have the same timestamp
    fit_info = get_fit_info_cfrl(fit, opt.sim_experiment{1});
    [fitpar, base, ext] = fileparts(fit_info.res_file);
    
    % put in the main experiment directory instead of the sim_experiment
    % subdirectories
    par = fullfile(info.model_dir, fit_info.model_type);
    if ~exist(par, 'dir')
        mkdir(par);
    end
end

% get subject(s) to process
subjnos = unique(data.subject);
if ~isempty(opt.subj_ind)
    subj_ind = opt.subj_ind;
    subjnos = subjnos(subj_ind);
    n_subj = length(subj_ind);
    outfile = fullfile(par, sprintf('%s_%s_%d.mat', base, res_name, subj_ind));
else
    n_subj = length(subjnos);
    subj_ind = 1:n_subj;
    outfile = fullfile(par, sprintf('%s_%s.mat', base, res_name));
end

% load existing results
if exist(outfile, 'file')
    if opt.overwrite
        fprintf('Overwriting existing results in: %s\n', outfile);
        delete(outfile);
    else
        fprintf('Loading existing results from: %s\n', outfile);
        load(outfile);
    end
else
    fprintf('Saving results to: %s\n', outfile);
end

% record context
if ~exist('c', 'var')
    disp('Recording context for best-fitting parameters...')
    
    % iterate over conditions, running each simulation based on the
    % best-fitting parameters for that condition
    if ~isempty(opt.sim_experiment)
        % store parameters and data in [subjects x condition] cell arrays
        n_subj = length(subj_ind);
        n_cond = length(opt.sim_experiment);
        cond_data = cell(n_subj, n_cond);
        cond_param = cell(n_subj, n_cond);
        
        % will store context in merged format
        s = struct;
        s.c = cell(1, n_cond);
        s.c_in = cell(1, n_cond);
        s.ic = cell(1, n_cond);
        
        % simulate each condition
        for i = 1:n_cond
            simdef = sim_def_cfrl(opt.sim_experiment{i}, fit);
            fit_info = get_fit_info_cfrl(fit, opt.sim_experiment{i});
            stats = getfield(load(fit_info.res_file, 'stats'), 'stats');
            [cond_data(:,i), cond_param(:,i), s.c{i}, s.c_in{i}, s.ic{i}] = ...
                indiv_context_cfrl(stats(subj_ind), simdef);
        end
        
        % get full data for each subject to get the correct list order
        subj_data = cell(n_subj, 1);
        for i = 1:n_subj
            subj_data{i} = trial_subset(data.subject==subjnos(i), data);
        end
        
        % fill in all lists in the original order in the full data struct
        ucond = unique(data.pres.distractor(:,1));
        f = struct;
        fnames = fieldnames(s);
        for i = 1:length(fnames)
            % initialize the full cell array for each subject
            full_pres = cell(1, n_subj);
            full_rec = cell(1, n_subj);
            for j = 1:n_subj
                full_pres{j} = cell(size(subj_data{j}.pres_itemnos));
                full_rec{j} = cell(size(subj_data{j}.rec_itemnos));
            end
            
            % aggregate lists across conditions
            for j = 1:n_cond
                con_pres = s.(fnames{i}){j}.pres;
                con_rec = s.(fnames{i}){j}.rec;
                for k = 1:n_subj
                    ind = subj_data{k}.pres.distractor(:,1) == ucond(j);
                    full_pres{k}(ind,:) = con_pres{k};
                    full_rec{k}(ind,1:size(con_rec{k},2)) = con_rec{k};
                end
            end
            f.(fnames{i}).pres = full_pres;
            f.(fnames{i}).rec = full_rec;
        end

        % unpack into separate variables
        c = f.c;
        c_in = f.c_in;
        ic = f.ic;
        subj_param = cond_param;
    else
        % all data simulated with the same parameters
        simdef = sim_def_cfrl(experiment, fit);
        stats = getfield(load(fit_info.res_file, 'stats'), 'stats');
        [subj_data, subj_param, c, c_in, ic] = ...
            indiv_context_cfrl(stats(subj_ind), simdef);
        cond_param = {};
    end
    
    % extract context subregion indices
    NL = size(stats(1).param.loc_vec, 1);
    NC = size(stats(1).param.cat_vec, 1);
    NS = size(stats(1).param.sem_vec, 1);
    subregion = struct('NL', NL, 'NC', NC, 'NS', NS);
    subregion.L = 1:NL;
    subregion.C = (NL + 1):(NL + NC);
    subregion.S = (NL + NC + 1):(NL + NC + NS);
    
    save(outfile, 'subj_data', 'subj_param', 'subregion', 'c', 'c_in', 'ic');
end

% decode EEG
pat_file = cell(1, n_subj);
for i = 1:n_subj
    switch experiment
      case 'cfr'
        filename = sprintf('psz_abs_emc_sh_rt_t2_LTP%03d.mat', subjnos(i));
        pat_file{i} = fullfile('~/work/cfr/eeg/study_patterns', filename);
      case 'cdcfr2'
        filename = sprintf('psz_rt_rpost_beta_CFR%03d.mat', subjnos(i));
        pat_file{i} = fullfile('~/work/cdcfr2/eeg/study_patterns', filename);
    end
end

if ~exist('eeg_evidence_raw', 'var')
    disp('Decoding EEG...')
    eeg_evidence_raw = cell(1, n_subj);
    eeg_perf = NaN(1, n_subj);
    for i = 1:n_subj
        pat = getfield(load(pat_file{i}, 'pat'), 'pat');
        [eeg_evidence_raw{i}, eeg_perf(i)] = decode_eeg(pat);
    end
    save(outfile, 'pat_file', 'eeg_evidence_raw', 'eeg_perf', '-append');
end

% match EEG to full set of trials
if ~exist('eeg_evidence', 'var')
    disp('Matching up trials...')
    eeg_evidence = cell(1, n_subj);
    for i = 1:n_subj
        n_trial = prod(size(subj_data{i}.pres_itemnos));
        eeg_evidence_exp = NaN(n_trial, 3);
        session = repmat(subj_data{i}.session, [1 24])';
        session = session(:);
        itemno = subj_data{i}.pres_itemnos';
        itemno = itemno(:);
        
        pat = getfield(load(pat_file{i}, 'pat'), 'pat');
        events = pat.dim.ev.mat;
        for j = 1:length(events)
            ind = session == events(j).session & itemno == events(j).itemno;
            if nnz(ind) == 0
                keyboard
            end
            eeg_evidence_exp(ind,:) = eeg_evidence_raw{i}(j,:);
        end
        eeg_evidence{i} = eeg_evidence_exp;
    end
    save(outfile, 'eeg_evidence', '-append');
end

% decode context without added noise
if ~exist('con_evidence_raw', 'var')
    disp('Decoding context...')
    con_evidence_raw = cell(1, n_subj);
    con_perf_raw = cell(1, n_subj);
    for i = 1:n_subj
        [con_evidence_raw{i}, con_perf_raw{i}] = ...
            decode_context(ic.pres{i}, subj_data{i}.pres.category);
    end
    save(outfile, 'con_evidence_raw', 'con_perf_raw', '-append');
end

% decode context with matched noise
if ~exist('con_evidence', 'var')
    disp('Decoding context with matched noise...')
    con_evidence_rep = cell(1, n_subj);
    con_evidence = cell(1, n_subj);
    con_perf = cell(1, n_subj);
    sigma = NaN(1, n_subj);
    for i = 1:n_subj
        keyboard
        [con_evidence_rep{i}, con_perf{i}, sigma(i)] = ...
            decode_context_match(ic.pres{i}, subj_data{i}.pres.category, ...
                                 eeg_perf(i), w, optim_opt);
        con_evidence{i} = mean(con_evidence_rep{i}, 3);        
    end
    save(outfile, 'con_evidence_rep', 'con_evidence', ...
         'con_perf', 'sigma', '-append');
end
