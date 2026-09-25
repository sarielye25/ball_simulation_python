import math
import random

mu_k = 0.3 # the coefficient of kinetic friction
mu_s = 0.4 # the coefficient of static friction
t1 = round(random.uniform(0.1, 1.0), 2) # the duration of the force applied to the cuboid
v0 = round(random.uniform(0, 10), 2) # the initial velocity of the cuboid, to be double checked.
t = 1 # the time step for the simulation

m = 1.0 # the mass of the cuboid
g = 9.81 # the gravity

def starting_from_rest(F, m, tr, tf):
    if not 0 <= tr <= tf:
        raise ValueError("Times must satisfy 0 <= tr <= tf.")

    if abs(F) <= mu_s * m * g:
        return 0.0, 0.0

    f = -math.copysign(1.0, F) * mu_k * m * g
    a1 = (F + f) / m
    v1 = a1 * tr
    d1 = 0.5 * a1 * tr**2

    return v1, d1

def after_removal_of_force(v1, tr, tf):
    if not 0 <= tr <= tf:
        raise ValueError("Times must satisfy 0 <= tr <= tf.")

    if v1 == 0:
        return 0.0, 0.0

    a2 = -math.copysign(1.0, v1) * mu_k * g
    ts = abs(v1) / abs(a2)
    t_left = min(ts, tf - tr)
    d2 = v1 * t_left + 0.5 * a2 * t_left**2

    if ts <= tf - tr:
        v = 0.0
    else:
        v = v1 + a2 * t_left

    return v, d2


def motion_cal(F, m, tr, tf, v0):
    if not 0 <= tr <= tf:
        raise ValueError("Times must satisfy 0 <= tr <= tf.")

    if v0 == 0:
        v1, d1 = starting_from_rest(F, m, tr, tf)
        v, d2 = after_removal_of_force(v1, tr, tf)
        return v, d1 + d2

    f = -math.copysign(1.0, v0) * mu_k * m * g
    a1 = (F + f) / m
    v1 = v0 + a1 * tr

    if v1 * v0 > 0:
        d1 = v0 * tr + 0.5 * a1 * tr**2
        v, d2 = after_removal_of_force(v1, tr, tf)
        return v, d1 + d2

    elif v1 * v0 == 0:
        d = v0 * tr + 0.5 * a1 * tr**2
        return 0.0, d

    else:
        ts = abs(v0) / abs(a1)
        d0 = v0 * ts + 0.5 * a1 * ts**2
        tr2 = tr - ts
        tf2 = tf - ts
        v1, d1 = starting_from_rest(F, m, tr2, tf2)
        v, d2 = after_removal_of_force(v1, tr2, tf2)
        return v, d0 + d1 + d2




            
        
