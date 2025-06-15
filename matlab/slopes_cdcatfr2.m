function slopes_cdcatfr2(slopes_file, out_file)

load(slopes_file, 'res')
category_names = {'curr', 'prev', 'base'};
condition_names = {'ifr', 'cd1', 'cd2'};

slope = [];
category = [];
condition = [];
for i = 1:length(category_names)
    for j = 1:length(condition_names)
        cat = category_names{i};
        cond = condition_names{j};
        x = squeeze(res.(cat).(cond).subj(2, 4, :));

        slope = [slope; x];
        category = [category; repmat(cat, size(x))];
        condition = [condition; repmat(cond, size(x))];
    end
end

tab = table(category, condition, slope);
writetable(tab, out_file);
