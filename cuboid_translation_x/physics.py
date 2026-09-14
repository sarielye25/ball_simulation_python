# This script is about the deterministic physics laws for the calculation of the final position given the selected applied force.
# The laws are written as functions that could be direcly imported and used in simulation script.

g = 9.8

def phy(F, m, t0, n, n1):
    if F < n1*m*g:
        print("The applied force is not enough to overcome the frictional force.")
        return 0
    else:
        d = (F - n1*m*g)/m*t0**2 + ((F - n*m*g)/m * t0)**2 / 2*n*g
        return d

# later with time-step updates, to track time.
