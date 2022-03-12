addpath ~/matlab/aperture/patclass
addpath ~/matlab/aperture/stats

patterns = [
   -0.1022   -1.2141    1.5442
   -0.2414   -1.1135    0.0859
    0.3192   -0.0068   -1.4916
    0.3129    1.5326   -0.7423
   -0.8649   -0.7697   -1.0616
   -0.0301    0.3714    2.3505
   -0.1649   -0.2256   -0.6156
    0.6277    1.1174    0.7481
    1.0933   -1.0891   -0.1924
    1.1093    0.0326    0.8886
   -0.8637    0.5525   -0.7648
    0.0774    1.1006   -1.4023
];

flattened = flatten_pattern(patterns);
flattened(1, 2) = nan;
flattened(3, 1) = nan;
flattened(3, 3) = nan;
flattened(7, 1) = nan;
missing = all(isnan(flattened), 2);
trimmed = flattened(~missing, :);
imputed = remove_nans(trimmed);
scaled = rescale(imputed);

fprintf('[\n')
for i = 1:size(scaled, 1)
    fprintf('    [%.4f, %.4f, %.4f],\n', scaled(i, :))
end
fprintf(']\n')
