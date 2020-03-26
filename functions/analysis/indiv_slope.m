function b = indiv_slope(mat)

b = NaN(size(mat, 1), 1);
for i = 1:size(mat, 1)
    p = polyfit(1:3, mat(i,1:3), 1);
    b(i) = p(1);
end
