function fixed = fix_item_column(tab, verbose)
%FIX_ITEM_COLUMN   Deal with missing words in the item column.
%
%  fixed = fix_item_column(tab, verbose)

if nargin < 2
    verbose = 0;
end

% get all item numbers across subjects
uitemno = unique(tab.item_number);

% remove intrusions
uitemno = uitemno(uitemno ~= -1);
for i = 1:length(uitemno)
    match = tab.item_number == uitemno(i);
    [c, ia, ib] = unique(tab.item(match));
    if length(c) > 1
        n = zeros(size(c));
        for j = 1:length(c)
            n(j) = nnz(ib == j);
        end

        [max_n, max_ind] = max(n);
        if verbose > 0
            fprintf('%s => %s\n', c{n ~= max_n}, c{max_ind})
        end
        tab.item(match) = c(max_ind);
    end
end
