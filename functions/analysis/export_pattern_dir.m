function export_pattern_dir(pattern_dir, filebase, out_dir)
    %EXPORT_PATTERN_DIR   Export all patterns in a directory.
    %
    %  export_pattern_dir(pattern_dir, filebase, out_dir)

    if ~exist(out_dir, 'dir')
        mkdir(out_dir)
    end
    files = dir(fullfile(pattern_dir, filebase));
    for i = 1:length(files)
        % load the pattern file
        s = load(fullfile(pattern_dir, files(i).name));

        % extract subject from the filename
        [par, name, ext] = fileparts(files(i).name);
        c = split(name, '_');
        subject = c{end};

        % set output file paths
        pattern_file = fullfile(out_dir, sprintf('sub-%s_pattern.txt', subject));
        events_file = fullfile(out_dir, sprintf('sub-%s_events.csv', subject));

        % convert pattern to samples x variables format and save
        x = s.pat.mat;
        xsize = size(x);
        y = reshape(x, [xsize(1) prod(xsize(2:end))]);
        y = double(y);
        save(pattern_file, 'y', '-ascii');

        % convert events to table and save to csv
        tab = events2table_cfr_study(s.pat.dim.ev.mat);
        writetable(tab, events_file);
    end
end
