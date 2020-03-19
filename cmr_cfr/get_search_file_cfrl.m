function file = get_search_file_cfrl(res_dir, model_type)
%GET_SEARCH_FILE_CFRL   Get file with search results.
%
%  file = get_search_file_cfrl(res_dir, model_type)

search_dir = fullfile(res_dir, model_type);

% get all mat-files
files = dir(fullfile(search_dir, '*.mat'));

if isempty(files)
    file = '';
    return
end

% find mat files with a date at the end
%pattern = [model_type '_\d{4}-\d{2}-\d{2}\d?\.mat'];
pattern = [model_type '_\d{8}T\d{6}\.mat'];
f = @(x) ~isempty(regexp(x, pattern));
match = cellfun(f, {files.name}) & ~[files.isdir];

% extract the date strings
search_files = {files(match).name};
%search_dates = cellfun(@(x) regexp(x, '\d{4}-\d{2}-\d{2}', 'match'), ...
%                       search_files);
search_dates = cellfun(@(x) regexp(x, '\d{8}T\d{6}', 'match'), ...
                       search_files);
%fprintf('Found %d search files.\n', length(search_files));

% convert to date number and sort
%datenums = datenum(search_dates, 'yyyy-mm-dd');
datenums = datenum(search_dates, 'yyyymmddTHHMMSS');
latest_num = max(datenums);

% files on most recent date
latest_files = search_files(datenums==latest_num);
sorted_latest = sort(latest_files);

file = fullfile(search_dir, sorted_latest{end});
%fprintf('Returning latest file: %s.\n', file);
