function opt = model_features_cfrl(model_type)
%MODEL_FEATURES_CFRL   Set non-numeric options for the model.
%
%  Given a model type string, determines the options to be
%  used. Each set of options defines a model variant.
%
%  opt = model_features_cfrl(model_type)

opt.sem_model = '';

if namecheck({'_qc' '_qi' '_qic'}, model_type)
    opt.qsem = true;
else
    opt.qsem = false;
end
if namecheck('_dc', model_type)
    opt.dc = true;
else
    opt.dc = false;
end
if namecheck('_loc', model_type)
    opt.loc = true;
else
    opt.loc = false;
end
if namecheck('_cat', model_type)
    opt.cat = true;
else
    opt.cat = false;
end
if namecheck('wikiw2v', model_type)
    opt.sem_model = 'wikiw2v';
    opt.sem = true;
else
    opt.sem = false;
end

opt.stop_rule = 'op';
opt.init_item = false;


function tf = namecheck(opt_name, model_type)

if ischar(opt_name)
    opt_name = {opt_name};
end

tf = false(1, length(opt_name));
for i = 1:length(opt_name)
    tf(i) = ~isempty(strfind(model_type, opt_name{i}));
end
tf = any(tf);
