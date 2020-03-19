function stats = fit_clust_stats(fits, experiments)
%FIT_CLUST_STATS   Clustering stats for different models.
%
%  stats = fit_clust_stats(fits, experiments)

stats = struct();
stats.temp = [];
stats.cat = [];
stats.sem = [];
for i = 1:length(fits)
    info = get_fit_info_cfrl(fits{i}, experiments{i});
    load(info.stat_file);
    files = get_exp_info_cfrl(experiments{i});
    if i == 1 || ~strcmp(experiments{i}, experiments{i-1})
        load(files.wikiw2v.mat);
    end
    
    % temporal clustering within category
    clean_mask = make_clean_recalls_mask2d(data.recalls);
    pres_mask = true(size(data.pres.category));
    temp = general_temp_fact(data.recalls, data.pres.category, ...
                             data.subject, 1, clean_mask, clean_mask, ...
                             pres_mask, pres_mask, false);
    stats.temp = [stats.temp temp];
    
    % category clustering
    cat = source_fact(data.rec.category, data.subject, ...
                      clean_mask, clean_mask);
    stats.cat = [stats.cat cat];
    
    % semantic clustering within category
    sem = sim_fact_cat(data.rec_itemnos, data.pres_itemnos, sem_mat, ...
                       data.pres.category, 1, data.subject);
    stats.sem = [stats.sem sem];
end
