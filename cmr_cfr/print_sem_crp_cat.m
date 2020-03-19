function print_sem_crp_cat(act, poss, category, sem_mat, edges, ...
                           centers, cat_type, fig_file)
%PRINT_SEM_CRP_CAT   Plot semantic CRP curve by category.
%
%  print_sem_crp_cat(act, poss, category, sem_mat, edges, centers,
%                    cat_type, fig_file)

x = centers;
y = NaN(3, length(centers));
l = NaN(3, length(centers));
u = NaN(3, length(centers));
for i = 1:3
    % uses implicit expansion, which requires R2016b or later
    switch cat_type
      case 1
        % within category
        mask = category==i & category'==i;
      case 2
        % from this category
        mask = category==i & category'~=i;
      case 3
        % to this category
        mask = category~=i & category'==i;
    end

    bin_crp = dist_item_crp(act, poss, sem_mat, 'edges', edges, ...
                            'mask', mask);
    mat = bin_crp(:,1:end-1);
    n = sum(~isnan(mat), 1);
    mat(:,n < 10) = NaN;
    y(i,:) = nanmean(mat);
    [l(i,:), u(i,:)] = bootstrap_ci(mat, 1, 5000, .05);
end

clf
h = mseb(x, y, cat(3, u-y, y-l));
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
l = legend([h.mainLine], {'celebrity' 'location' 'object'});
set(l, 'Location', 'NorthWest', font_prop{:}, 'Box', 'off')

print(gcf, '-depsc', fig_file);
