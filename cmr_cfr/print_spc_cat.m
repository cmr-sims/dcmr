function print_spc_cat(data, fig_file, analysis)
%PRINT_SPC_CAT   Plot serial position curves for each category.
%
%  print_spc_cat(data, fig_file, analysis)

if nargin < 3
    analysis = 'spc';
end

switch analysis
  case 'spc'
    f = @spc;
  case 'pfr'
    f = @pfr;
  otherwise
    error('Unknown analysis: %s', analysis)
end

% calculate each spc and sem over subjects
clean_mask = make_clean_recalls_mask2d(data.recalls);
x = 1:data.listLength;
y = NaN(3, data.listLength);
l = NaN(3, data.listLength);
u = NaN(3, data.listLength);
for i = 1:3
    pres_mask = data.pres.category == i;
    rec_mask = (data.rec.category == i) & clean_mask;
    p_recall = f(data.recalls, data.subject, data.listLength, ...
                 rec_mask, pres_mask);
    y(i,:) = mean(p_recall, 1);
    [l(i,:), u(i,:)] = bootstrap_ci(p_recall, 1, 5000, .05);
    %err(i,:) = std(p_recall, [], 1) / sqrt(size(p_recall, 1)-1);
end

% plot using error regions
clf
h = mseb(x, y, cat(3, u-y, y-l));
%h = mseb(x, y, err);
a = gca;
set(a, 'YLim', [0 1], 'YTick', 0:.2:1, ...
       'XLim', [0 25], 'XTick', [1 4:4:24]);
xlabel('serial position')
ylabel('recall probability')
box off
font_prop = {'FontSize' 28 'FontWeight' 'normal' ...
             'FontName' 'Helvetica'};
set(a, 'LineWidth', 1, font_prop{:})
set(get(a, 'XLabel'), font_prop{:})
set(get(a, 'YLabel'), font_prop{:})
l = legend([h.mainLine], {'celebrity' 'location' 'object'});
set(l, 'Location', 'NorthWest', font_prop{:}, 'Box', 'off')

% print to file
print(gcf, '-depsc', fig_file);
