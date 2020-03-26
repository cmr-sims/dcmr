function print_crp_cat(data, cat_type, lag_type, fig_file)
%PRINT_CRP_CAT   Plot serial position curves for each category.
%
%  print_crp_cat(data, cat_type, lag_type, fig_file)

% calculate each crp and sem over subjects
clean_mask = make_clean_recalls_mask2d(data.recalls);
max_lag = 5;
x = -max_lag:max_lag;
x_full = -(data.listLength-1):(data.listLength-1);
x_ind = find(ismember(x_full, x));
y = NaN(3, length(x));
l = NaN(3, length(x));
u = NaN(3, length(x));
for i = 1:3
    pres_mask = true(size(data.pres.category));
    rec_mask = clean_mask;
    
    switch cat_type
      case 1
        % include only transitions within this category
        from_mask_pres = pres_mask & data.pres.category == i;
        from_mask_rec = rec_mask & data.rec.category == i;

        to_mask_pres = from_mask_pres;
        to_mask_rec = from_mask_rec;
      case 2
        % get transitions starting at this category, and transitioning
        % to any other category
        from_mask_pres = pres_mask & data.pres.category == i;
        from_mask_rec = rec_mask & data.rec.category == i;
        
        to_mask_pres = pres_mask & data.pres.category ~= i;
        to_mask_rec = rec_mask & data.rec.category ~= i;
      case 3
        % get transitions starting at some other category, and
        % transitioning to this category
        from_mask_pres = pres_mask & data.pres.category ~= i;
        from_mask_rec = rec_mask & data.rec.category ~= i;
        
        to_mask_pres = pres_mask & data.pres.category == i;
        to_mask_rec = rec_mask & data.rec.category == i;
      case 4
        % get transitions starting at this category, and
        % transitioning to any item
        from_mask_pres = pres_mask & data.pres.category == i;
        from_mask_rec = rec_mask & data.rec.category == i;
        to_mask_pres = pres_mask;
        to_mask_rec = rec_mask;
    end
    
    switch lag_type
      case 'cat'
        % cut out lags excluded because of cat_type
        lag_crps = cat_crp(data.recalls, data.pres.category, ...
                           data.rec.category, data.subject, ...
                           cat_type, ...
                           from_mask_rec, to_mask_rec, ...
                           from_mask_pres, to_mask_pres);
      case 'standard'
        % use normal lag
        lag_crps = crp(data.recalls, data.subject, data.listLength, ...
                       from_mask_rec, to_mask_rec, ...
                       from_mask_pres, to_mask_pres);
    end
    inc_lag_crps = lag_crps(:,x_ind);
    y(i,:) = mean(inc_lag_crps, 1);
    [l(i,:), u(i,:)] = bootstrap_ci(inc_lag_crps, 1, 5000, .05);
    %err(i,:) = std(p_recall, [], 1) / sqrt(size(p_recall, 1)-1);
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
set(a, 'YLim', [0 .4], 'YTick', 0:.1:.4)
xlabel('lag')
ylabel('conditional recall probability')
box off
font_prop = {'FontSize' 28 'FontWeight' 'normal' ...
             'FontName' 'Helvetica'};
set(a, 'LineWidth', 1, font_prop{:})
set(get(a, 'XLabel'), font_prop{:})
set(get(a, 'YLabel'), font_prop{:})
l = legend([h1.mainLine], {'celebrity' 'location' 'object'});
set(l, 'Location', 'NorthWest', font_prop{:}, 'Box', 'off')

% print to file
print(gcf, '-depsc', fig_file);
