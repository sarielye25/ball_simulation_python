# Training Results

This file records diagnostic analysis of each run of each training protocol.
The diagnosis format is:

```text 
Status: success/failure [if failure, add stop reason]
SUN: [stopping_update_number]
Observation: [what i see from the data]
PR: [possible reason for results] 
OD: [optimzation direction, both from human and ai]
```

## First Training Protocol


### Run 20260929_142936_371616
```text
Status: failure [ no_progress ]
SUN: 20
Observation: Absolute losses of both v and d prediction are big. Though the loss curve decreases rapidly, the pass rate still gets low. I saw the standardized mse getting below 0.001, but at 20th update, still 7997 samples failed as intermediate level, which is strange.
PR: 
OD: 1. check qt demonstration system, verify the results.
```

PR:
1. The patience caused early stopping. Maybe prolong the updates could help better train the model.
2. The learning-rate schedule. Too rapid to decrease; the reduction is too big.
3. Neural network's structure. Maybe 4-32-32-4 layer is not enough.
4. Standardization balances output scales, but average squared error does not directly optimize the fraction of samples satisfying two strict physical tolerances. Large-error samples can dominate learning while many small physical errors still fail the accuracy criterion.

OD:

1. Conduct controlled experience to see if prolong lr reduction patience and decrease the changing value would help.
2. Test if more or less hidden layer would help improve the results.

## First Training Controlled Experiences

### experiment on learning rate
Results:

1. There is improvement when we prolong the lr reduction patience and lower the reduction value. But that only helps reduce the general mse at 10^-3 level.
2. There is more improvement when the model goes through more rounds of updates. However, as the update number increases, the impact of it decreases dramatically. When we add updates from 4000 to 10000, the improvement on general mse is only 10^-4 level.
   
Analysis:

1. These are the not the major cause to the bad training of the model. I must find other elements

Future Experiments:

1. Combine E1 and E2, use E1's lr schedule and run more updates again. Call it E3.
2. Add one and two hidden layers to the neural network, call it E4.
