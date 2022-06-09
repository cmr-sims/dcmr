function tab = events2table_cfr_study(events)
%EVENTS2TABLE_CFR_STUDY   Convert CFR study events to table format.
%
%  tab = events2table_cfr_study(data)

    % index columns
    n_event = length(events);
    subject = NaN(n_event, 1);
    for i = 1:n_event
        subject(i) = str2num(events(i).subject(end-2:end));
    end
    session = [events.session]' - 1;
    trial_type = repmat({'study'}, [n_event, 1]);
    position = [events.serialpos]';
    n_trial = 16;
    list_raw = [events.trial]';
    list = NaN(n_event, 1);
    for i = 1:n_event
        list(i) = list_raw(i) + (n_trial * (session(i) - 1));
    end

    % items and responses
    item = {events.item}';
    item_index = [events.itemno]' - 1;
    category_idx = [events.category]' + 1;
    category = cell(n_event, 1);
    category(category_idx == 1) = {'cel'};
    category(category_idx == 2) = {'loc'};
    category(category_idx == 3) = {'obj'};
    response = [events.resp]';
    response_time = [events.rt]' / 1000;

    % list type
    list_type = cell(n_event, 1);
    list_type([events.listtype] == 3) = {'mixed'};
    list_type([events.listtype] < 3) = {'pure'};
    list_category = cell(n_event, 1);
    list_category([events.listtype] == 0) = {'cel'};
    list_category([events.listtype] == 1) = {'loc'};
    list_category([events.listtype] == 2) = {'obj'};
    list_category([events.listtype] == 3) = {'mixed'};

    tab = table(subject, list, position, trial_type, item, item_index, ...
                session, list_type, category, response, response_time, ...
                list_category);
end
