function print_sem_crp_distract(data, pool, sem, bin, fig_file)
%PRINT_SEM_CRP_DISTRACT   Plot semantic CRP by distraction condition.
%
%  print_sem_crp_distract(data, pool, sem, bin, fig_file)

split = split_distract_cfrl(data);
mask = {true(length(pool.category), length(pool.category)) ...
        pool.category == pool.category' ...
        pool.category ~= pool.category'};

% tally up transitions between all items, separately for each
% distraction condition
act = cell(1, length(split));
poss = cell(1, length(split));
for i = 1:length(split)
    [act{i}, poss{i}] = item_crp(split{i}.recalls, ...
                                 split{i}.pres_itemnos, ...
                                 split{i}.subject, ...
                                 length(sem.sem_mat));
end

min_samp = 5;
[parent, filename, ext] = fileparts(fig_file);
mask_name = {'' 'within' 'between'};
for i = 1:length(mask)
    % for each transition type, plot all distraction conditions
    x = repmat(bin.centers, [3 1]);
    y = NaN(3, length(bin.centers));
    l = NaN(3, length(bin.centers));
    u = NaN(3, length(bin.centers));
    for j = 1:length(split)
        [bin_crp, act_crp] = dist_item_crp(act{j}, poss{j}, sem.sem_mat, ...
                                           'edges', bin.edges, ...
                                           'mask', mask{i});
        mat = bin_crp(:,1:end-1);
        n = sum(act_crp(:,1:end-1) > min_samp, 1);
        if size(mat, 1) > 1
            mat(:,n < min_samp) = NaN;
        end
        y(j,:) = nanmean(mat, 1);
        [l(j,:), u(j,:)] = bootstrap_ci(mat, 1, 5000, .05);
    end
    
    % plot using error regions
    clf
    h = mseb(x, y, cat(3, u-y, y-l));
    %h = mseb(x, y, err);
    a = gca;
    set(a, 'XLim', [.5 1], 'XTick', .5:.1:1, ...
           'YLim', [0 .2], 'YTick', 0:.05:.2)
    xlabel('semantic similarity')
    ylabel('conditional response probability')
    box off
    font_prop = {'FontSize' 28 'FontWeight' 'normal' ...
                 'FontName' 'Helvetica'};
    set(a, 'LineWidth', 1, font_prop{:})
    set(get(a, 'XLabel'), font_prop{:})
    set(get(a, 'YLabel'), font_prop{:})
    l = legend([h.mainLine], {'IFR' 'CDS' 'CDL'});
    set(l, 'Location', 'NorthWest', font_prop{:}, 'Box', 'off')
    
    if ~isempty(mask_name{i})
        cat_file = fullfile(parent, sprintf('%s_%s%s', filename, ...
                                            mask_name{i}, ext));
    else
        cat_file = fullfile(parent, sprintf('%s%s', filename, ext));
    end
    print(gcf, '-depsc', cat_file);
end
