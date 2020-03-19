function prep_data_cdcfr2()

data_file = '~/work/cdcfr2/stat_data_cdcatfr2.mat';
wiki_w2v_file = '~/work/cdcfr2/models/mat_wiki_w2v.mat';

proj_dir = fileparts(mfilename('fullpath'));
data_dir = fullfile(proj_dir, 'data');

% data struct for cdcatfr2, with 10 participants
load(data_file);

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

% change distractor field from ms to s
data.pres.distractor = data.pres.distractor / 1000;

% remove fields that aren't needed for simulations
data = rmfield(data, {'subjid' 'times' 'intrusions'});
f = {'period' 'type' 'eegfile' 'eegoffset' 'artifactMS' 'serialpos' ...
     'resp' 'rt' 'recalled' 'finalrecalled' 'mstime' 'msoffset' ...
     'subject' 'session' 'trial' 'item' 'itemno' 'rectime' 'intrusion'};
data.pres = rmfield(data.pres, f);
data.rec = rmfield(data.rec, f);

save(fullfile(data_dir, 'cdcfr2_data.mat'), 'data');
data_full = data;

% clean the recall matrices (remove repeats and intrusions)
data = clean_frdata(data);
save(fullfile(data_dir, 'cdcfr2_data_clean.mat'), 'data')
data_clean = data;

% save out separate files for each distraction condition
distlen = [0 2.5 7.5];
for i = 1:length(distlen)
    data = trial_subset(data_clean.pres.distractor(:,1) == distlen(i), ...
                        data_clean);
    filename = sprintf('cdcfr2_data_clean_d%d.mat', i - 1);
    save(fullfile(data_dir, filename), 'data');
end

% category
[itemno, ind] = unique(data_clean.pres_itemnos);
category = data_clean.pres.category(ind);
item = data_clean.pres_items(ind);
save(fullfile(data_dir, 'cdcfr2_pool.mat'), 'item', 'itemno', 'category');

% semantics
load(wiki_w2v_file)
sem_file = fullfile(data_dir, 'cdcfr2_wikiw2v_raw.mat');
sem_mat = squareform(1 - squareform(rdm));
save(sem_file, 'sem_mat', 'rdm', 'vectors', 'items');

% scaled version
sem_file = fullfile(data_dir, 'cdcfr2_wikiw2v.mat');
sem_mat = squareform(rescale(squareform(sem_mat)));
vectors = zscore(vectors, [], 2);
save(sem_file, 'sem_mat', 'rdm', 'vectors', 'items');
