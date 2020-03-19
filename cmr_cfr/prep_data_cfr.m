function prep_data_cfr()

data_file = '~/work/cfr/stat_data_cfr.mat';
wiki_w2v_file = '~/work/cfr/models/mat_wiki_w2v.mat';

proj_dir = fileparts(mfilename('fullpath'));
data_dir = fullfile(proj_dir, 'data');

% data struct for catFR_LTP, with all 40 original participants
load(data_file);

% remove familarization session, which has a different size
data = rmfield(data, 'fam');

% build new category matrices with 1,2,3 as category codes instead
% of 0,1,2. This makes analysis easier and avoids having a category
% code that is the same as the padding used for recall matrices
pres_cat = zeros(size(data.pres.category));
rec_cat = zeros(size(data.rec.category));
cat_codes = [0 1 2];
mask = data.recalls ~= 0;
for i = 1:length(cat_codes)
    pres_cat(data.pres.category==cat_codes(i)) = i;
    rec_cat(data.rec.category==cat_codes(i) & mask) = i;
end

% some recalls have NaN for category
rec_cat(isnan(data.rec.category)) = NaN;

data.pres.category = pres_cat;
data.rec.category = rec_cat;

% remove fields that aren't needed for simulations, and that take
% up extra space on disk (a problem when running many replications
% of a simulation)
data = rmfield(data, {'subjid' 'times' 'intrusions'});
f = {'period' 'type' 'eegfile' 'eegoffset' 'artifactMS' 'serialpos' ...
     'resp' 'rt' 'recalled' 'finalrecalled' 'mstime' 'msoffset' ...
     'subject' 'session' 'trial' 'item' 'itemno' 'rectime' 'intrusion'};
data.pres = rmfield(data.pres, f);
data.rec = rmfield(data.rec, f);

save(fullfile(data_dir, 'cfr_data.mat'), 'data');
data_full = data;

% included EEG subjects (taken from ACCRE, under results/catFR_a2/exp.mat)
inc_ids = {'LTP001' 'LTP002' 'LTP003' 'LTP005' 'LTP008' 'LTP011' 'LTP016' 'LTP018' 'LTP022' 'LTP023' 'LTP024' 'LTP025' 'LTP027' 'LTP028' 'LTP029' 'LTP031' 'LTP032' 'LTP033' 'LTP034' 'LTP035' 'LTP037' 'LTP038' 'LTP040' 'LTP041' 'LTP042' 'LTP043' 'LTP044' 'LTP045' 'LTP046'};

inc_nos = NaN(1, length(inc_ids));
for i = 1:length(inc_ids)
    inc_nos(i) = str2num(inc_ids{i}(4:end));
end

% data struct for 29 EEG participants
data = trial_subset(ismember(data_full.subject, inc_nos), data_full);
save(fullfile(data_dir, 'cfr_eeg_data.mat'), 'data');
data_eeg = data;

% data struct for mixed lists for 29 EEG participants
data = trial_subset(data_eeg.pres.listtype(:,1)==3, data_eeg);
save(fullfile(data_dir, 'cfr_eeg_mixed_data.mat'), 'data');
data_eeg_mixed = data;

% clean the recall matrices (remove repeats and intrusions)
data = clean_frdata(data);
save(fullfile(data_dir, 'cfr_eeg_mixed_data_clean.mat'), 'data')

% category
[itemno, ind] = unique(data.pres_itemnos);
category = data.pres.category(ind);
item = data.pres_items(ind);
save(fullfile(data_dir, 'cfr_pool.mat'), 'item', 'itemno', 'category');

% semantics
load(wiki_w2v_file)
sem_file = fullfile(data_dir, 'cfr_wikiw2v_raw.mat');
sem_mat = squareform(1 - squareform(rdm));
save(sem_file, 'sem_mat', 'rdm', 'vectors', 'items');

% scaled version
sem_file = fullfile(data_dir, 'cfr_wikiw2v.mat');
sem_mat = squareform(rescale(squareform(sem_mat)));
vectors = zscore(vectors, [], 2);
save(sem_file, 'sem_mat', 'rdm', 'vectors', 'items');
