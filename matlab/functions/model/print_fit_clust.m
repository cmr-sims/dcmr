function print_fit_clust(mat, clust_type, fit_names, fig_file)
%PRINT_FIT_CLUST   Make a bar plot of clustering by model.
%
%  print_fit_clust(mat, clust_type, fit_names, fig_file)

clf
hold on
m = mean(mat);
sem = std(mat) / sqrt(size(mat, 1)-1);
%x = [1 3:9];
x = 1:length(fit_names);
isdata = strcmp(fit_names, 'data');

[hbar, herr] = ebar(x(~isdata), m(~isdata), sem(~isdata));
hbar.FaceColor = [.8 .8 .8];
hbar.LineStyle = 'none';
hbar.BarWidth = .8;
herr.CapSize = 15;

[hbar, herr] = ebar(x(isdata), m(isdata), sem(isdata));
hbar.FaceColor = [.6 .6 .6];
hbar.LineStyle = 'none';
hbar.BarWidth = .8;
herr.CapSize = 15;

set(gca, 'XTick', x, 'XTickLabel', fit_names);
box off
switch clust_type
  case 'temp'
    ylabel('Temporal clustering')
    chance = .5;
  case 'cat'
    ylabel('Category clustering')
    chance = 1/3;
  case 'sem'
    ylabel('Semantic clustering')
    chance = .5;
end

plot([0 x(end)+1], [chance chance], '--k', 'LineWidth', 1);
set(gca, 'YLim', [.3 .7], 'YTick', .3:.1:.7)
set(gca, 'FontSize', 16, 'LineWidth', 1)

if ~isempty(fig_file)
    set(gcf, 'PaperPosition', [0 0 4 4])
    print(gcf, '-depsc', fig_file);
end
