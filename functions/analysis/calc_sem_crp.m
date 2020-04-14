function res = calc_sem_crp(act, poss, sem_mat, edges, centers, ...
                            varargin)
%CALC_SEM_CRP   Calculate semantic CRP curve.
%
%  calc_sem_crp(act, poss, sem_mat, edges, centers, fig_file, ...)
%
%  OPTIONS:
%  mask - [items x items] logical array - true(size(sem_mat))
%      Item pairs to include in analysis.

def.mask = true(size(sem_mat));
opt = propval(varargin, def);

[bin_crp, act_crp] = dist_item_crp(act, poss, sem_mat, 'edges', edges, ...
                                  'mask', opt.mask);

min_samp = 5;
mat = bin_crp(:,1:end-1);
n = sum(act_crp(:,1:end-1) > min_samp, 1);
if size(mat, 1) > 1
    mat(:,n < min_samp) = NaN;
end
x = centers;
y = nanmean(mat, 1);
[l, u] = bootstrap_ci(mat, 1, 5000, .05);

% the subject level transition probabilities
res.bin_crp = bin_crp;
% the subject level counts
res.act_crp = act_crp;
% the bin centers
res.xvals = x;
% the mean transition probabilities across subjects that will be plotted 
res.yvals = y;
% bootstrap estimated confidence intervals
res.lower_ci = l;
res.upper_ci = u;

