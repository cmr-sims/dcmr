function print_clust_cat(data, fig_file)
%PRINT_CLUST_CAT   Plot clustering by category.
%
%  print_clust_cat(data, fig_file)

clean_mask = make_clean_recalls_mask2d(data.recalls);
p_within = NaN(length(unique(data.subject)), 3);
for j = 1:3
    from_mask = clean_mask & data.rec.category == j;
    p_within(:,j) = source_fact(data.rec.category, data.subject, ...
                                from_mask, clean_mask);
end

colors = get(groot, 'defaultAxesColorOrder');

[l, u] = bootstrap_ci(p_within, 1, 5000, .05);

clf
x = 1:3;
y = nanmean(p_within, 1);

htemp = errorbar(x, y, y-l, u-y);
y_lim = get(gca, 'YLim');
delete(htemp)

for i = 1:3
    hold on
    [hbar(i), herr(i)] = ebar(x(i), y(i), l(i), u(i));
    hbar(i).FaceColor = colors(i,:);
    hbar(i).LineStyle = 'none';
    herr(i).CapSize = 20;
end
a = gca;
plot([0 4], [1/3 1/3], '--k', 'LineWidth', 1)
set(a, 'YLim', [0 .7], 'YTick', 0:.1:.7, ...
       'XLim', [0 4], 'XTick', 1:3, 'XTickLabel', ...
       {'celebrity' 'landmark' 'object'})
font_prop = {'FontSize' 26 'FontWeight' 'normal' ...
             'FontName' 'Helvetica'};
set(a, 'LineWidth', 1, font_prop{:})
set(get(a, 'XLabel'), font_prop{:})
set(get(a, 'YLabel'), font_prop{:})
ylabel('within-category probability')
box off
print(gcf, '-depsc', fig_file);
