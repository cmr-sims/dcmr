function [aic, waic] = aic_subj(logl, n, V)
%AIC_SUBJ   AIC and AIC weights by subject.
%
%  [aic, waic] = aic_subj(logl, n, V)
%
%  INPUTS
%  logl - [subjects x models] numeric array
%      Log likelihood for each subject, for each model.
%
%  n - [subjects x 1] numeric array
%      Number of data points fit for each subject.
%
%  V - [1 x models] numeric array
%      Number of parameters in each model.
%
%  OUTPUTS
%  aic - [subjects x models] numeric array
%      Akaike Information Criterion for each subject and model.
%
%  waic - [subjects x models] numeric array
%      AIC weights for each subject.

% AIC
[n_subj, n_model] = size(logl);
aic = NaN(n_subj, n_model);
for i = 1:n_subj
    for j = 1:n_model
        aic(i,j) = -2*logl(i,j) + 2*V(j) + (2*V(j)*(V(j)+1)) / (n(i)-V(j)-1);
    end
end

% AIC weights
waic = NaN(n_subj, n_model);
for i = 1:n_subj
    daic = aic(i,:) - min(aic(i,:));
    for j = 1:n_model
        waic(i,j) = exp(-.5 * daic(j)) / sum(exp(-.5 * daic));
    end
end
