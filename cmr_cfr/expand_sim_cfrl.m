function net_data = expand_sim_cfrl(seq, data, n_rep)
%EXPAND_SIM_CFRL   Get a full data struct from simulated data.
%
%  net_data = expand_sim_cfrl(seq, data, n_rep)

if nargin < 3
    n_rep = 1;
end

net_data = rmfield(data, 'recalls_vec');
if n_rep > 1
    net_data = repmat_struct(net_data, n_rep);
end
net_data.recalls = seq;

% update recall fields that can be derived from study data (e.g. itemno)
net_data = update_derived_fields(net_data);


function s_full = repmat_struct(s, n_rep)

    s_full = s;
    f = fieldnames(s);
    for i = 1:length(f)
        if isstruct(s.(f{i}))
            s_full.(f{i}) = repmat_struct(s.(f{i}), n_rep);
        elseif ~isscalar(s.(f{i}))
            s_full.(f{i}) = repmat(s.(f{i}), n_rep, 1);
        end
    end

function data = update_derived_fields(data)

    if isfield(data, 'rec_itemnos') && isfield(data, 'pres_itemnos')
        data.rec_itemnos = study_mat2recall_mat(data.pres_itemnos, ...
                                                data.recalls, 0, NaN);
    end
    if isfield(data, 'rec_items') && isfield(data, 'pres_items')
        data.rec_items = study_mat2recall_mat(data.pres_items, data.recalls, ...
                                              {}, {});
    end
    if isfield(data, 'rec') && isfield(data, 'pres')
        if isfield(data.rec, 'category') && isfield(data.pres, 'category')
            data.rec.category = study_mat2recall_mat(data.pres.category, ...
                                                     data.recalls, 0, NaN);
        end
        
        if isfield(data.rec, 'listtype') && isfield(data.pres, 'listtype')
            data.rec.listtype = study_mat2recall_mat(data.pres.listtype, ...
                                                     data.recalls, 0, NaN);
        end
    end
