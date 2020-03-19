function evid = mean_evidence(res)
%MEAN_EVIDENCE   Mean evidence for the correct category.
%
%  evid = mean_evidence(res)

evid = NaN(1, length(res.iterations));
for i = 1:length(res.iterations)
    evid(i) = mean(res.iterations(i).acts(res.iterations(i).targs==1));
end
evid = mean(evid);
