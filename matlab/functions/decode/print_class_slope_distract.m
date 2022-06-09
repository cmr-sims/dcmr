function print_class_slope_distract(mat, fig_file)
%PRINT_CLASS_SLOPE_DISTRACT   Plot classifier slope by category type.
%
%  The mat should be [ctype x distract x subject].
%
%  print_class_slope_distract(mat, fig_file)

m = mean(mat, 3);
[l, u] = bootstrap_ci(mat, 3, 5000, 0.05);

x = 1:size(mat, 2);
[hbar, herr] = ebar([], m, l, u);

colors = [.8 .8 .8
          .5 .5 .5
          .1 .1 .1];
for i = 1:length(hbar)
    hbar(i).FaceColor = colors(i,:);
    hbar(i).LineStyle = 'none';
    hbar(i).BarWidth = .8;
    herr(i).CapSize = 10;
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
