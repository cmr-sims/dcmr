function print_crp_distract(data, fig_file)
%PRINT_CRP_DISTRACT   Plot log-CRP by distraction.
%
%  print_crp_distract(data, fig_file)

split = split_distract_cfrl(data);

max_lag = 5;
x = -max_lag:max_lag;
x_full = -(data.listLength-1):(data.listLength-1);
x_ind = find(ismember(x_full, x));
y = NaN(3, length(x));
l = NaN(3, length(x));
u = NaN(3, length(x));

for i = 1:length(split)
    lag_crps = crp(split{i}.recalls, split{i}.subject, data.listLength);
    inc_lag_crps = lag_crps(:,x_ind);
    y(i,:) = mean(inc_lag_crps, 1);
    [l(i,:), u(i,:)] = bootstrap_ci(inc_lag_crps, 1, 5000, .05);
end

% plot using error regions
clf
hold on
inc = x < 0;
h1 = mseb(x(inc), y(:,inc), cat(3, u(:,inc)-y(:,inc), y(:,inc)-l(:,inc)));
inc = x > 0;
h2 = mseb(x(inc), y(:,inc), cat(3, u(:,inc)-y(:,inc), y(:,inc)-l(:,inc)));
%h = mseb(x, y, err);
a = gca;
set(a, 'XLim', [-(max_lag+1) (max_lag+1)], 'XTick', x)
xlabel('lag')
ylabel('conditional recall probability')
box off
font_prop = {'FontSize' 28 'FontWeight' 'normal' ...
             'FontName' 'Helvetica'};
set(a, 'LineWidth', 1, font_prop{:})
set(get(a, 'XLabel'), font_prop{:})
set(get(a, 'YLabel'), font_prop{:})
l = legend([h1.mainLine], {'IFR' 'CDS' 'CDL'});
set(l, 'Location', 'NorthWest', font_prop{:}, 'Box', 'off')

% print to file
if ~isempty(fig_file)
    print(gcf, '-depsc', fig_file);
end
