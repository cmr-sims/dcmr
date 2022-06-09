function write_benchmark_cfr(raw_file, out_file)
%WRITE_BENCHMARK_CFR   Write the benchmark dataset to csv format.
%
%  write_benchmark_cfr(raw_file, out_file)

data = getfield(load(raw_file, 'data'), 'data');
tab = frdata2table_cfr(data);
include = tab.subject == 1 & strcmp(tab.list_type, 'mixed');
subj = tab(include, :);
writetable(subj, out_file)
