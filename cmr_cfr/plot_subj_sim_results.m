function stats = plot_subj_sim_results(res)

data = res.fstruct.data;
param = check_param_cfrl(unpack_param(res.parameters, ...
                                      res.fstruct.param_info, res.fstruct));

% record context after presentation of each item and before each
% recall
param.record = {'c'};
[logl, logl_all, net] = logl_tcm(param, data);
assert(abs(-res.fitness - nansum(logl(:))) < .001, ...
       'Difference between fit and simulation.');

figure(1)
clf
plot_logl_all(logl_all, data.recalls, [5 6], [10 500]);

% run a generative simulation for analysis
seq = gen_tcm(param, data, 10);
simdata = expand_sim_cfrl(seq, data, 10);

figure(2)
clf
lag_crps_sim = crp(simdata.recalls, simdata.subject, simdata.listLength);
lag_crps_data = crp(data.recalls, data.subject, data.listLength);
plot_crp({lag_crps_data lag_crps_sim});

figure(3)
clf
plot_fr_summary(data.recalls, data.subject, data.listLength);

figure(4)
clf
plot_fr_summary(simdata.recalls, simdata.subject, simdata.listLength);

lbc_data = lbc(data.pres.category, data.rec.category, data.subject, ...
               'recall_mask', make_clean_recalls_mask2d(data.recalls));
lbc_sim = lbc(simdata.pres.category, simdata.rec.category, simdata.subject, ...
               'recall_mask', make_clean_recalls_mask2d(simdata.recalls));
fprintf('LBC data: %.4f\n', lbc_data);
fprintf('LBC sim:  %.4f\n', lbc_sim);

stats = struct;
stats.data = data;
stats.simdata = simdata;
stats.logl = logl;
stats.logl_all = logl_all;
stats.net = net;
