
eeg_dir = '/Users/morton/Dropbox/work/cmr_cfr/cfr/eeg/study_patterns';
pat_file = fullfile(eeg_dir, 'psz_abs_emc_sh_rt_t2_LTP001.mat');

pat = getfield(load(pat_file, 'pat'), 'pat');
tic
[evidence, perf] = decode_eeg(pat);
toc
