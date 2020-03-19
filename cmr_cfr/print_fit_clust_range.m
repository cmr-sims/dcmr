function print_fit_clust_range(mat, clust_type, fit_names, fig_file)
%PRINT_FIT_CLUST_RANGE   Plot model clustering with a range for the data.
%
%  print_fit_clust_range(mat, clust_type, fit_names, fig_file)

isdata = strcmp(fit_names, 'data');

clf
hold on

% means for models
m = mean(mat(:,~isdata));
x = 1:length(fit_names(~isdata));

% 95% CI for data
[l, u] = bootstrap_ci(mat(:,isdata), 1, 5000, .05);

% bar plots with correct model color (models must be in a standard
% order)
for i = 1:length(m)
    hbar(i) = bar(x(i), m(i));
end

% colors = [.4078 .4392 .6980
%           .7216 .4314 .4235
%           .7961 .7804 .4157
%           .5922 .4314 .6706
%           .5529 .7490 .5804
%           .7412 .5451 .4078
%           .4627 .4627 .4745];
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
end

% data range
x_lim = [0 length(fit_names)];
plot(x_lim, [l l], '-k', 'LineWidth', 1);
plot(x_lim, [u u], '-k', 'LineWidth', 1);

set(gca, 'XTick', x, 'XTickLabel', fit_names(~isdata));
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

% chance line
plot([0 x(end)+1], [chance chance], '--k', 'LineWidth', 1);
set(gca, 'YLim', [0 1], 'YTick', 0:.2:1)
set(gca, 'FontSize', 14, 'LineWidth', 1)

if ~isempty(fig_file)
    set(gcf, 'PaperPosition', [0 0 4 4])
    print(gcf, '-depsc', fig_file);
end
