import math
import random


mu_k = 0.3
mu_s = 0.4
m = 1.0
g = 9.81
t_d = 1.0


def _advance_phase(F, m, mu_s, mu_k, g, t_d, v_0):
    if t_d == 0:
        return 0.0, v_0

    if v_0 == 0:
        if abs(F) <= mu_s * m * g:
            return 0.0, 0.0
        direction = math.copysign(1.0, F)
    else:
        direction = math.copysign(1.0, v_0)

    a1 = F / m - direction * mu_k * g

    if v_0 * a1 < 0:
        t_s = -v_0 / a1
        if t_s <= t_d:
            d_stop = v_0 * t_s / 2
            d_remaining, v = _advance_phase(
                F, m, mu_s, mu_k, g,
                t_d - t_s, 0.0,
            )
            return d_stop + d_remaining, v

    d = v_0 * t_d + 0.5 * a1 * t_d**2
    v = v_0 + a1 * t_d
    return d, v


def calculate_motion(
    F, m, mu_s, mu_k, g, t_f, t_d, v_0,
):
    """Return signed displacement (m) and final velocity (m/s) after push then release."""
    values = (
        F, m, mu_s, mu_k, g, t_f, t_d, v_0,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("All inputs must be finite numbers.")
    if m <= 0 or g <= 0:
        raise ValueError("Mass and gravity must be positive.")
    if not 0 <= mu_k <= mu_s:
        raise ValueError("Friction must satisfy 0 <= mu_k <= mu_s.")
    if not 0 <= t_f <= t_d:
        raise ValueError("Durations must satisfy 0 <= t_f <= t_d.")

    d_push, v_release = _advance_phase(
        F, m, mu_s, mu_k, g, t_f, v_0,
    )
    d_coast, v = _advance_phase(
        0.0, m, mu_s, mu_k, g, t_d - t_f, v_release,
    )
    return d_push + d_coast, v


if __name__ == "__main__":
    t_f = round(random.uniform(0.1, t_d), 2)
    v_0 = round(random.uniform(-10, 10), 2)
    F = 10.0
    d, v = calculate_motion(
        F, m, mu_s, mu_k, g, t_f, t_d, v_0,
    )
    print(f"Initial velocity: {v_0} m/s; force: {F} N; duration: {t_f} s")
    print(f"Displacement: {d:.6f} m; final velocity: {v:.6f} m/s")

