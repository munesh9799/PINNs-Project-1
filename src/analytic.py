import numpy as np

def sdof_params(m, c, k):
    """Natural frequency, damping ratio, damped frequency."""
    omega_n = np.sqrt(k / m)
    zeta = c / (2.0 * np.sqrt(k * m))
    omega_d = omega_n * np.sqrt(1.0 - zeta**2)
    return omega_n, zeta, omega_d


def sdof_response(t, m, c, k, x0, v0):
    """
    Analytical displacement response of an underdamped SDOF system.

    m x'' + c x' + k x = 0
    """
    omega_n, zeta, omega_d = sdof_params(m, c, k)

    A = x0
    B = (v0 + zeta * omega_n * x0) / omega_d

    x = np.exp(-zeta * omega_n * t) * (
        A * np.cos(omega_d * t) + B * np.sin(omega_d * t)
    )

    return x


def sdof_velocity(t, m, c, k, x0, v0):
    """Analytical velocity response."""
    omega_n, zeta, omega_d = sdof_params(m, c, k)

    A = x0
    B = (v0 + zeta * omega_n * x0) / omega_d

    exp_term = np.exp(-zeta * omega_n * t)

    v = exp_term * (
        -zeta * omega_n * (A * np.cos(omega_d * t) + B * np.sin(omega_d * t))
        + (-A * omega_d * np.sin(omega_d * t) + B * omega_d * np.cos(omega_d * t))
    )

    return v
