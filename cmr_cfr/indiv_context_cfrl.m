function [subj_data, subj_param, c, c_in, ic] = indiv_context_cfrl(stats, simdef)
%INDIV_CONTEXT_CFRL   Record states of context for subject simulations.
%
%  [subj_data, c, c_in] = indiv_context_cfrl(stats, simdef)

pool = load(simdef.pool_file);
subj_data = cell(1, length(stats));
subj_param = cell(1, length(stats));
c.pres = cell(1, length(stats));
c.rec = cell(1, length(stats));
c_in.pres = cell(1, length(stats));
c_in.rec = cell(1, length(stats));
for i = 1:length(stats)
    % construct parameters from the best-fitting vector
    param = unpack_param(stats(i).parameters, stats(i).param_info);
    param = propval(simdef.fixed, param, 'strict', false);
    param = prep_param_cfrl(param, simdef, pool.category);
    param = check_param_cfrl(param);

    % enable context recording
    param.record = {'c' 'c_in'};
    [logl, logl_all, net] = logl_tcm(param, stats(i).data);

    subj_data{i} = stats(i).data;
    subj_param{i} = param;
    c.pres{i} = net.pres.c;
    c.rec{i} = net.rec.c;
    c_in.pres{i} = net.pres.c_in;
    c_in.rec{i} = net.rec.c_in;
end

% zip together item and context
ic.pres = zip(c.pres, c_in.pres);
ic.rec = zip(c.rec, c_in.rec);


function ic = zip(c, c_in)

  ic = cell(size(c));
  for i = 1:length(c)
      ic{i} = cell(size(c{i}));
      for j = 1:numel(c{i})
          ic{i}{j} = cat(1, c_in{i}{j}, c{i}{j});
      end
  end
  