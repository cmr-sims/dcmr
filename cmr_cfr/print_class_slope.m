function print_class_slope(mat, fig_file)
%PRINT_CLASS_SLOPE   Plot classifier slope by category type.
%
%  print_class_slope(mat, fig_file)

m = mean(mat, 1);
[l, u] = bootstrap_ci(mat, 1, 5000, 0.05);

x = 1:size(mat, 2);

hold on
for i = 1:length(m)
    [hbar(i), herr(i)] = ebar(x(i), m(i), l(i), u(i));
end

for i = 1:length(hbar)
    hbar(i).FaceColor = [.8 .8 .8];
    hbar(i).LineStyle = 'none';
    hbar(i).BarWidth = .8;
    herr(i).CapSize = 15;
end

set(gca, 'XTick', x, 'XTickLabel', {'curr' 'prev' 'base'});
ylabel('evidence slope')
set(gca, 'YLim', [-.015 .015], 'YTick', -.015:.005:.05)
set(gca, 'FontSize', 16, 'LineWidth', 1)
box off

if ~isempty(fig_file)
    set(gcf, 'PaperPosition', [0 0 4.5 4])
    print(gcf, '-depsc', fig_file);
end
