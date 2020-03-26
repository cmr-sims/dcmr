function tab = frdata2table_cfr(data)
%FRDATA2TABLE_CFR   Convert CFR data to table format.
%
%  tab = frdata2table_cfr(data)

% study event onsets relative to the first presentation in each trial
data.pres.onset = (data.pres.mstime - data.pres.mstime(:, 1)) / 1000;

% recall event onsets relative to the start of the recall period
times = data.times;
times(times == 0) = NaN;
data.rec.onset = times / 1000;

% run the conversion
extra = {'session', 'listtype', 'category', 'resp', 'rt'};
names = {'', 'list_type', '', 'response', 'response_time'};
if isfield(data, 'fam')
    data = rmfield(data, 'fam');
end
tab = frdata2table(data, extra, names);

% adjust field format
tab.session = tab.session - 1;
tab.category = tab.category + 1;
tab.response_time = tab.response_time / 1000;

% split list information into type and category
list_type = cell(size(tab, 1), 1);
list_type(tab.list_type == 3) = {'mixed'};
list_type(tab.list_type < 3) = {'pure'};

list_cat = cell(size(tab, 1), 1);
list_cat(tab.list_type == 0) = {'cel'};
list_cat(tab.list_type == 1) = {'loc'};
list_cat(tab.list_type == 2) = {'obj'};
list_cat(tab.list_type == 3) = {'mixed'};

tab.list_type = list_type;
tab.list_category = list_cat;

category = cell(size(tab, 1), 1);
category(tab.category == 1) = {'cel'};
category(tab.category == 2) = {'loc'};
category(tab.category == 3) = {'obj'};
tab.category = category;

% deal with one subject who's item strings are cut off at one word;
% was probably some problem with the parsing data
tab = fix_item_column(tab);
