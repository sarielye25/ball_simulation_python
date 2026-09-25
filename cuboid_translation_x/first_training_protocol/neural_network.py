"""Predict [d, v] from [v0, F, t1, t] using a 4 -> 32 -> 32 -> 2 network.

t1 is the force duration; t is the observation time, with 0 <= t1 <= t.
Inputs have shape (batch_size, 4); outputs have shape (batch_size, 2).
"""

from torch import nn


class transmodel(nn.Module):
    # The parameters and bias for each neuron is initialized thourgh nn.Linear
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, X):
        predictions = self.linear_relu_stack(X)
        return predictions
