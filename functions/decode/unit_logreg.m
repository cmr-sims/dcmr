addpath ~/matlab/pmvpa/core/learn
addpath ~/matlab/pmvpa/core/util

patterns = [
   -0.1022   -1.2141    1.5442
   -0.2414   -1.1135    0.0859
    0.3192   -0.0068   -1.4916
    0.3129    1.5326   -0.7423
   -0.8649   -0.7697   -1.0616
   -0.0301    0.3714    2.3505
   -0.1649   -0.2256   -0.6156
    0.6277    1.1174    0.7481
    1.0933   -1.0891   -0.1924
    1.1093    0.0326    0.8886
   -0.8637    0.5525   -0.7648
    0.0774    1.1006   -1.4023
];
targets = [
    1 0 0
    1 0 0
    0 1 0
    0 1 0
    0 0 1
    0 0 1
    1 0 0
    1 0 0
    0 1 0
    0 1 0
    0 0 1
    0 0 1
];
chunks = [1 1 1 1 1 1 2 2 2 2 2 2];
train = chunks == 1;
test = chunks == 2;

options = struct('penalty', 10);
scratchpad = train_logreg(patterns(train, :)', targets(train, :)', options);
[prob, scratchpad] = test_logreg(patterns(test, :)', targets(test, :)', scratchpad);
prob'

out = logRegFun(targets(train, 1)', patterns(train, :)', 10, 1e-4, 5000);
out
