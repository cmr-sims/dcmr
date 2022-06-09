function print_clust_distract(data, fig_file, analysis)
%PRINT_CLUST_DISTRACT   Plot clustering by distraction.
%
%  print_clust_distract(data, fig_file, analysis)

split = split_distract_cfrl(data);

clust = NaN(length(unique(data.subject)), 3);
for i = 1:length(split)
    rec_mask = make_clean_recalls_mask2d(split{i}.recalls);
    switch analysis
      case 'lbc'
        clust(:,i) = lbc(split{i}.pres.category, split{i}.rec.category, ...
                         split{i}.subject, 'recall_mask', rec_mask);
      case 'tempfact'
        pres_mask = true(size(split{i}.pres.category));
        clust(:,i) = ...
            general_temp_fact(split{i}.recalls, split{i}.pres.category, ...
                              split{i}.subject, 1, rec_mask, rec_mask, ...
                              pres_mask, pres_mask, false);
      otherwise
        error('Unknown analysis: %s', analysis);
    end
end

x = 1:3;
y = nanmean(clust, 1);
[l, u] = bootstrap_ci(clust, 1, 5000, .05);

clf
hold on

colors = get(groot, 'defaultAxesColorOrder');
htemp = errorbar(x, y, y-l, u-y);
y_lim = get(gca, 'YLim');
delete(htemp)

for i = 1:3
    [hbar(i), herr(i)] = ebar(x(i), y(i), l(i), u(i));
    hbar(i).FaceColor = colors(i,:);
    hbar(i).LineStyle = 'none';
    herr(i).CapSize = 20;
end
a = gca;
plot([0 4], [1/3 1/3], '--k', 'LineWidth', 1)
set(a, 'XLim', [0 4], 'XTick', 1:3, 'XTickLabel', ...
       {'IFR' 'CDS' 'CDL'})
switch analysis
  case 'lbc'
    set(a, 'YLim', [1 2.5], 'YTick', 1:.5:2.5);
    ylabel('LBC_{sem}')
  case 'tempfact'
    set(a, 'YLim', [.6 .75], 'YTick', .6:.05:.75)
    ylabel('Temporal score')
end
font_prop = {'FontSize' 26 'FontWeight' 'normal' ...
             'FontName' 'Helvetica'};
set(a, 'LineWidth', 1, font_prop{:})
set(get(a, 'XLabel'), font_prop{:})
set(get(a, 'YLabel'), font_prop{:})

if ~isempty(fig_file)
    print(gcf, '-depsc', fig_file);
end
