function print_crp(data, fig_file)
%PRINT_CRP_CAT   Plot serial position curves for each category.
%
%  print_crp(data, cat_type, lag_type, fig_file)

% lags to display
max_lag = 5;
x = -max_lag:max_lag;
x_full = -(data.listLength-1):(data.listLength-1);
x_ind = find(ismember(x_full, x));

% calculate CRP
lag_crps = crp(data.recalls, data.subject, data.listLength);

% statistics
inc_lag_crps = lag_crps(:,x_ind);
y = mean(inc_lag_crps, 1);
[l, u] = bootstrap_ci(inc_lag_crps, 1, 5000, .05);

% plot using error regions
clf
hold on
lineopt = struct('col', {{[0 0 0]}});
inc = x < 0;
h1 = mseb(x(inc), y(inc), cat(3, u(inc)-y(inc), y(inc)-l(inc)), lineopt);
inc = x > 0;
h2 = mseb(x(inc), y(inc), cat(3, u(inc)-y(inc), y(inc)-l(inc)), lineopt);
a = gca;
set(a, 'XLim', [-(max_lag+1) (max_lag+1)], 'XTick', x)
set(a, 'YLim', [0 .4], 'YTick', 0:.1:.4)
xlabel('lag')
ylabel('conditional recall probability')
box off
font_prop = {'FontSize' 28 'FontWeight' 'normal' ...
             'FontName' 'Helvetica'};
set(a, 'LineWidth', 1, font_prop{:})
set(get(a, 'XLabel'), font_prop{:})
set(get(a, 'YLabel'), font_prop{:})

% print to file
if ~isempty(fig_file)
    print(gcf, '-depsc', fig_file);
end
