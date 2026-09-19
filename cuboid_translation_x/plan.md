#  Stage One: Neural Network Training

## Goal of the simulation

Predicting the displacement and velocity of a cuboid within \delta t, the time interval, given its current velocity and the force applied. The movement of cuboid is one dimensional, only following the x-axis. 

The starting status of cuboid is unknown, it could be static or moving, with velocity given.

The push duration should be a variable. In training process and real prediction tests, it is a given information. It varies from 0.1s to 1s. The model should learn how to predict with force duration changing.

Environmental variables, such as gravity, coefficients of kinematic friction & static friction are fixed.

The input variables are (v_0, F, t_f), the output is (d, v_f)

## Preparation of labels
Labels should cover all possible situations. 

They are produced by label_preparation.py.
