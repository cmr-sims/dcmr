
addpath('/Users/morton/PycharmProjects/psifr/matlab');

data_file = '/Users/morton/matlab/cmr_bak/cmr_branch/fr/TFRLTP/data/peers_e1_data.mat';
load(data_file);

subs = trial_subset(data.listtype == -1, data, 1);
tab = frdata2table(subs);
tab.session = ceil(tab.list / 4);

out_file = '/Users/morton/Dropbox/work/cmr_cfr/peers/peers_notask.csv';
writetable(tab, out_file);
