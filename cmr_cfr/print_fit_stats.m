function print_fit_stats(mat, fit_names, fig_file)
%PRINT_FIT_STATS   Make a bar plot of fit statistics by model.
%
%  print_fit_clust(mat, fit_names, fig_file)

clf
m = mean(mat);
[l, u] = bootstrap_ci(mat, 1, 5000, 0.05);
%x = [1 3:9];
x = 1:length(fit_names);

hold on
for i = 1:length(m)
    [hbar(i), herr(i)] = ebar(x(i), m(i), l(i), u(i));
end

% colors = [.4078 .4392 .6980
%           .7216 .4314 .4235
%           .7961 .7804 .4157
%           .5922 .4314 .6706
%           .5529 .7490 .5804
%           .7412 .5451 .4078
%           .4627 .4627 .4745];
% pastel 1 with 6 categories from ColorBrewer
colors = ([179,205,227
           251,180,174
           255,255,204
           222,203,228
           204,235,197
           254,217,166
           200,200,200] / 256) -.1;

for i = 1:length(hbar)
    %hbar.FaceColor = [.8 .8 .8];
    hbar(i).FaceColor = colors(i,:);
    hbar(i).LineStyle = 'none';
    hbar(i).BarWidth = .8;
    herr(i).CapSize = 15;
end

% [hbar, herr] = ebar(x, m, l, u);
% hbar.FaceColor = [.8 .8 .8];
% hbar.LineStyle = 'none';
% hbar.BarWidth = .8;
% herr.CapSize = 15;

set(gca, 'XTick', x, 'XTickLabel', fit_names);
ylabel('AIC weight')
set(gca, 'YLim', [0 1], 'YTick', 0:.2:1)
set(gca, 'FontSize', 16, 'LineWidth', 1)
box off

if ~isempty(fig_file)
    set(gcf, 'PaperPosition', [0 0 4.5 4])
    print(gcf, '-depsc', fig_file);
end
