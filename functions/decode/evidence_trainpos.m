function [eeg_evid, con_evid, n] = evidence_trainpos(eeg, con, category)
%EVIDENCE_TRAINPOS_CAT   Classifier evidence by trainpos and category.
%
%  Given EEG classifier evidence and context classifier evidence for one
%  subject, calculate mean evidence as a function of train position,
%  category type, and category. This can then be used to calculate category
%  integration rate.
%
%  [eeg_evid, con_evid, n] = evidence_trainpos_cat(eeg, con, category)
%
%  INPUTS
%  eeg - [trials x categories] matrix
%      Classifier evidence for EEG data.
%
%  con - [trials x categories] matrix
%      Classifier evidence for states of context from the model.
%
%  category - [lists x items] matrix
%      Stimulus category. Sorting order must match the column order in
%      eeg and con.
%
%  OUTPUTS
%  eeg_evid - [trainpos x category types] matrix
%      Classifier evidence for EEG by train position, category type
%      (current, previous, baseline), and category.
%
%  con_evid - [trainpos x category types] matrix
%      Classifier evidence statistics for context.
%
%  n - [trainpos x category types] matrix
%      Number of trials in each bin.

% for each train after the first, get the current, previous, and
% baseline category
[curr, prev, base, trainpos] = train_category(category);

include = ~any(isnan(eeg), 2);
ucat = unique(category);
n_cat = length(ucat);
utrainpos = unique(trainpos);
n_trainpos = length(utrainpos);

vec_cat = unwrap(category);
vec_trainpos = unwrap(trainpos);

ctypes = {'curr' 'prev' 'base'};
ctype.curr = unwrap(curr);
ctype.prev = unwrap(prev);
ctype.base = unwrap(base);
n_ctypes = length(ctypes);

% for each trial, get each type of evidence
n_rep = size(con, 3);
n_trial = size(eeg, 1);
eeg_ctype = NaN(n_trial, 3);
con_ctype = NaN(n_trial, 3, n_rep);
for i = 1:n_trial
    for j = 1:n_ctypes
        col = ctype.(ctypes{j})(i);
        if ~isnan(col)
            eeg_ctype(i,j) = eeg(i,col);
            con_ctype(i,j,:) = con(i,col,:);
        end
    end
end

% average within train position bins
eeg_evid = NaN(n_trainpos, n_ctypes);
con_evid = NaN(n_trainpos, n_ctypes, n_rep);
n = NaN(n_trainpos, n_ctypes);
for i = 1:n_trainpos
    for j = 1:n_ctypes
        % for this ctype (curr, prev, base), get all trials with the
        % correct train position. Exclude trials where the EEG is
        % undefined (i.e., signal was marked as bad for that trial)
        include = vec_trainpos == utrainpos(i) & ...
                  ~isnan(ctype.(ctypes{j})) & ...
                  ~any(isnan(eeg), 2);
        
        % mean evidence for the category of interest
        eeg_evid(i,j) = mean(eeg_ctype(include,j), 1);
        con_evid(i,j,:) = mean(con_ctype(include,j,:), 1);
        n(i,j) = nnz(include);
    end
end


function y = unwrap(x)

y = x';
y = y(:);
