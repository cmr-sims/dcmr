function tab = frdata2table_cdcfr2(data)
%FRDATA2TABLE_CDCFR2   Convert CDCFR2 data to table format.
%
%  tab = frdata2table_cdcfr2(data)

% study event onsets relative to the first presentation in each trial
data.pres.onset = (data.pres.mstime - data.pres.mstime(:, 1)) / 1000;

% recall event onsets relative to the start of the recall period
times = data.times;
times(times == 0) = NaN;
data.rec.onset = times / 1000;

% run the conversion
extra = {'itemno', 'session', 'distractor', 'category', 'resp', 'rt'};
names = {'item_number', '', 'distractor', '', 'response', 'response_time'};
tab = frdata2table(data, extra, names);

% adjust field format
tab.session = tab.session - 1;
tab.category = tab.category + 1;
tab.response_time = tab.response_time / 1000;
tab.distractor = tab.distractor / 1000;

category = cell(size(tab, 1), 1);
category(tab.category == 1) = {'cel'};
category(tab.category == 2) = {'loc'};
category(tab.category == 3) = {'obj'};
tab.category = category;

% adjust distractor field to apply to whole list, so intrusions will
% be included too when filtering for those lists
usubject = unique(tab.subject);
for i = 1:length(usubject)
    ulist = unique(tab(tab.subject == usubject(i), :).list);
    for j = 1:length(ulist)
        list_ind = tab.subject == usubject(i) & tab.list == ulist(j);
        distractor = unique(tab(list_ind, :).distractor);
        distractor = distractor(~isnan(distractor));
        if length(distractor) > 1
            error('Multiple distractor values in a list.')
        end
        tab{list_ind, 'distractor'} = distractor;
    end
end
