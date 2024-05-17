function prep_data_cfr_cymr(data_file, out_file)
%PREP_DATA_CFR_CYMR   Prepare CFR data for simulation using cymr.
%
%  prep_data_cfr_cymr(data_file, out_file)

data = getfield(load(data_file, 'data'), 'data');
tab = frdata2table_cfr(data);
inc_nos = [1, 2, 3, 5, 8, 11, 16, 18, 22, 23, 24, 25, 27, 28, 29, ...
           31, 32, 33, 34, 35, 37, 38, 40, 41, 42, 43, 44, 45, 46];
tab_eeg = tab(ismember(tab.subject, inc_nos), :);
tab_eeg_mixed = tab_eeg(strcmp(tab_eeg.list_type, 'mixed'), :);
writetable(tab_eeg_mixed, out_file);
