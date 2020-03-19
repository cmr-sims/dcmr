
%% load basic data

files = get_exp_info_cfrl('cdcfr2');
real = load(files.data);
sem = load(files.wikiw2v.raw);

create_sem_crp_bins(real.data, sem.sem_mat, files.wikiw2v.bin);

%% plot SPC for all subjects

usubj = unique(real.data.subject);
udistract = unique(real.data.pres.distractor);
for i = 1:length(usubj)
    s_data = trial_subset(real.data.subject == usubj(i), real.data);
    p_recall = [];
    for j = 1:length(udistract)
        d_data = trial_subset(s_data.pres.distractor(:,1) == udistract(j), ...
                              s_data);
        p_recall = [p_recall; spc(d_data.recalls, d_data.subject, ...
                                  d_data.listLength)];
    end
    subplot(3,4,i)
    h = plot(p_recall');
    set(gca, 'YLim', [0 1]);
    if i == 1
        l = legend(h, {'IFR' 'CDS' 'CDL'});
        l.Location = 'NorthWest';
    end
end

%% sample simulation to test out distraction code

% fit of one subject
files = get_exp_info_cfrl('cdcfr2-2');
real = load(files.data);
res = indiv_search_cfrl('cdcfr2-2', 'base', 'search_type', 'de');
[res_file, stats] = save_search_cfrl(res, 'cdcfr2-2', 'base');
data = run_indiv_best_params_cfrl('cdcfr2-2', 'base', 'n_rep', 100);

split_net = split_distract_cfrl(data);
split_real = split_distract_cfrl(real.data);
p_recall_net = NaN(length(split), 24);
p_recall_real = NaN(length(split), 24);
for i = 1:length(split)
    rec = spc(split_net{i}.recalls, split_net{i}.subject, ...
              split_net{i}.listLength);
    p_recall_net(i,:) = rec;
    
    rec = spc(split_real{i}.recalls, split_real{i}.subject, ...
              split_real{i}.listLength);
    p_recall_real(i,:) = rec;
end

subplot(2,1,1)
plot(p_recall_real');
set(gca, 'YLim', [0 1]);
subplot(2,1,2)
plot(p_recall_net');
set(gca, 'YLim', [0 1]);

%% test out effects of distraction

strength = [1 2 4 8 16];
strength = strength ./ sum(strength);
fp1 = @(s, T) (s.^T) ./ sum(s.^T);
fp2 = @(s, T) exp((2*s)/T) / (sum(exp(2*s)./T));
f = {fp1 fp2};
t = linspace(0.01, 1, 5);
for i = 1:length(f)
    subplot(1,2,i);
    hold on
    for j = 1:length(t)
        plot(f{i}(strength, t(j)));
    end
    set(gca, 'YLim', [0 1])
end

% does exponential scaling lead to sensitivity to distraction?
strength = [1 2 4 8 16];
strength = strength ./ sum(strength);
scale = linspace(0.01, 1, 5);
fp1 = @(s, T) (s.^T) ./ sum(s.^T);
fp2 = @(s, T) exp((2*s)/T) / (sum(exp(2*s)./T));
f = {fp1 fp2};
T = .5;
for i = 1:length(f)
    subplot(1,2,i);
    hold on
    for j = 1:length(scale)
        plot(f{i}(strength * scale(j), T));
    end
    set(gca, 'YLim', [0 1])
end

%% load first searches of different disruption parameters

res.base = load('/Users/morton/work/cdcfr2/tcm/tcm_dc_loc_cat_wikiw2v/tcm_dc_loc_cat_wikiw2v_20190120T131330.mat');
res.dsl = load('/Users/morton/work/cdcfr2/tcm/tcm_dc_loc_cat_wikiw2v_dsl/tcm_dc_loc_cat_wikiw2v_dsl_20190120T131330.mat');
res.dsc = load('/Users/morton/work/cdcfr2/tcm/tcm_dc_loc_cat_wikiw2v_dsc/tcm_dc_loc_cat_wikiw2v_dsc_20190120T131330.mat');
res.dsd = load('/Users/morton/work/cdcfr2/tcm/tcm_dc_loc_cat_wikiw2v_dsd/tcm_dc_loc_cat_wikiw2v_dsd_20190120T131330.mat');

base = [res.base.stats.fitness]';
dsl = [res.dsl.stats.fitness]';
dsc = [res.dsc.stats.fitness]';
dsd = [res.dsd.stats.fitness]';

t = table(base, dsl, dsc, dsd);


%% work on setting up faster iteration of distraction models

% load one subject
experiment = 'cdcfr2-2';
fit = 'local_cat_wikiw2v';
files = get_exp_info_cfrl(experiment);
real = load(files.data);

% base model fit based on likelihood
res = indiv_search_cfrl(experiment, fit, 'search_type', 'de_fast');
search = unpack_search_cfrl(res, experiment, fit);

simdef = sim_def_cfrl(experiment, fit);
data = search.data;
pool = load(simdef.pool_file);
custom = prep_param_cfrl(search.param, simdef, pool.category);

%% test out different RDMs

% add plotting functions for rdms
addpath ~/analysis/wikisim/ana

files = get_exp_info_cfrl('cdcfr2');
sem = load(files.wikiw2v.mat);
load(files.data);
sdata = trial_subset(data.subject==201, data);

pres_itemnos = sdata.pres_itemnos';
itemnos_vec = pres_itemnos(:);

pres_itemnos1 = [nan(size(sdata.pres_itemnos, 1), 1) ...
                 sdata.pres_itemnos(:,2:end)];
itemnos_vec1 = pres_itemnos1(:);

pres_category = sdata.pres.category';
category_vec = pres_category(:);

pres_session = repmat(sdata.session', [sdata.listLength 1]);
session_vec = pres_session(:);

pres_list = repmat(1:size(sdata.pres_itemnos, 1), [sdata.listLength 1]);
list_vec = pres_list(:);

pres_serialpos = repmat([1:sdata.listLength]', ...
                        [1 size(sdata.pres_itemnos, 1)]);
serialpos_vec = pres_serialpos(:);

f_exact = @(x,y) double(~all(x==y, 2));
model = struct();
model.session = squareform(pdist(session_vec, f_exact));
model.list = squareform(pdist(list_vec, f_exact));
model.serialpos = squareform(pdist(serialpos_vec, f_exact));
model.sem = sem.rdm(itemnos_vec, itemnos_vec);
model.cat = squareform(pdist(category_vec, f_exact));
model.item = squareform(pdist(itemnos_vec, f_exact));

f = fieldnames(model);
for i = 1:length(f)
    subplot(2,3,i)
    plot_rdm(model.(f{i}), 'prctile_range', [0 100]);
    title(f{i}, 'FontSize', 18)
end

%% what if we just fit the full standard model to each distraction condition
%% individually?

res_dir = '~/work/cdcfr2/figs';
fits = repmat({'local_cat_wikiw2v'}, [1 3]);
experiments = {'cdcfr2_d0' 'cdcfr2_d1' 'cdcfr2_d2'};
for i = 1:length(fits)
    fig_dir = fullfile(res_dir, strrep(experiments{i}, 'cdcfr2', 'model'));
    if ~exist(fig_dir, 'dir')
        mkdir(fig_dir)
    end
    print_fit_summary(fits{i}, experiments{i}, fig_dir);
end

% actual data for comparison
for i = 1:length(experiments)
    fig_dir = fullfile(res_dir, strrep(experiments{i}, 'cdcfr2', 'data'));
    if ~exist(fig_dir, 'dir')
        mkdir(fig_dir)
    end
    print_fit_summary('data', experiments{i}, fig_dir);
end

%% EEG decoding

% test run with one participant
sim_experiment = {'cdcfr2_d0' 'cdcfr2_d1' 'cdcfr2_d2'};
decode_cfrl('cdcfr2', 'local_cat_wikiw2v', 'decode_ic', [.3 1], ...
            'sim_experiment', sim_experiment, ...
            'subj_ind', 1, 'overwrite', true);

% testing out subregion weighting (or omission)
sim_experiment = {'cdcfr2_d0' 'cdcfr2_d1' 'cdcfr2_d2'};
decode_cfrl('cdcfr2', 'local_cat_wikiw2v', 'decode_ic_rtest', [.3 1], ...
            'sim_experiment', sim_experiment, ...
            'subj_ind', 1, 'overwrite', true);

fig_dir = '~/work/cdcfr2/figs/integ';
w = [.1 .3 .5];
for i = 1:length(w)
    name = sprintf('%.0f', w(i) * 100);
    res_name = sprintf('decode_ic_evid_test_%s', name);
    s = load_decode_cfrl('cdcfr2', 'local_cat_wikiw2v', ...
                         'decode_ic_evid_test_10');
    n_subj = length(s.c.pres);
    m_eeg = cell(1, n_subj);
    m_con = cell(1, n_subj);
    n = cell(1, n_subj);
    for j = 1:n_subj;
        [m_eeg{j}, m_con{j}, n{j}] = ...
            evidence_trainpos(s.eeg_evidence{j}, s.con_evidence{j}, ...
                              s.subj_data{j}.pres.category);
    end

    clf
    out_name = sprintf('integ_ic_evid_test_%s_eeg.eps', name);
    print_evid_trainpos(cat(3, m_eeg{:}), fullfile(fig_dir, out_name));
    clf
    out_name = sprintf('integ_ic_evid_test_%s_con.eps', name);
    print_evid_trainpos(cat(3, m_con{:}), fullfile(fig_dir, out_name));
end

s = load_decode_cfrl('cdcfr2', 'local_cat_wikiw2v', ...
                     'decode_ic_evid_test_100');
n_subj = length(s.c.pres);
m_eeg = cell(1, n_subj);
m_con = cell(1, n_subj);
n = cell(1, n_subj);
for j = 1:n_subj
    [m_eeg{j}, m_con{j}, n{j}] = ...
        evidence_trainpos(s.eeg_evidence{j}, s.con_evidence{j}, ...
                          s.subj_data{j}.pres.category);
end

clf
out_name = 'integ_ic_evid_test_100_eeg.eps';
print_evid_trainpos(cat(3, m_eeg{:}), fullfile(fig_dir, out_name));
clf
out_name = 'integ_ic_evid_test_100_con.eps';
print_evid_trainpos(cat(3, m_con{:}), fullfile(fig_dir, out_name));

% load distraction classification and plot
res_name = 'decode_ic_evid_wtest_50';
s = load_decode_cfrl('cdcfr2', 'cdcfr2_d0', 'local_cat_wikiw2v', res_name);
res = class_slope_distract(s);

fig_dir = '~/work/cdcfr2/figs/integ_distract_wtest';
if ~exist(fig_dir, 'dir')
    mkdir(fig_dir)
end

for i = 1:length(res.distract);
    clf
    out_name = sprintf('%s_d%.0f_eeg.eps', res_name, i - 1);
    print_evid_trainpos(squeeze(res.eeg_m(:,:,i,:)), ...
                        fullfile(fig_dir, out_name));
    clf
    out_name = sprintf('%s_d%.0f_con.eps', res_name, i - 1);
    print_evid_trainpos(squeeze(res.con_m(:,:,i,:)), ...
                        fullfile(fig_dir, out_name));
end

clf
out_name = sprintf('%s_slope_eeg.eps', res_name);
print_class_slope_distract(res.eeg_b, fullfile(fig_dir, out_name));
clf
out_name = sprintf('%s_slope_con.eps', res_name);
print_class_slope_distract(res.con_b, fullfile(fig_dir, out_name));
