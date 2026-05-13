import numpy as np

def numerical_grad_U(U, r, h=1e-5):
    """
    Numerically compute grad U(r) in 2D using central differences.

    Parameters
    ----------
    U : callable
        Scalar potential function U(x, y).
    r : array-like, shape (2,)
        Position vector [x, y].
    h : float
        Finite-difference step size.

    Returns
    -------
    grad : np.ndarray, shape (2,)
        Numerical gradient [dU/dx, dU/dy].
    """

    x, y = r

    dU_dx = (U(x + h, y) - U(x - h, y)) / (2 * h)
    dU_dy = (U(x, y + h) - U(x, y - h)) / (2 * h)

    return np.array([dU_dx, dU_dy], dtype=float)

def simulate_helmholtz_rayleigh_2d_cartesian(
    U,
    n_steps=50000,
    dt=1e-3,
    beta=1.0,
    v0=1.0,
    D=0.05,
    r0=np.array([0.0, 0.0]),
    v_init=np.array([0.1, 0.0]),
    grad_h=1e-5,
    seed=42,
):

    rng = np.random.default_rng(seed)

    r = np.zeros((n_steps, 2), dtype=float)
    v = np.zeros((n_steps, 2), dtype=float)
    speed = np.zeros(n_steps, dtype=float)
    time = np.arange(n_steps) * dt

    r[0] = np.asarray(r0, dtype=float)
    v[0] = np.asarray(v_init, dtype=float)
    speed[0] = np.linalg.norm(v[0])

    for i in range(1, n_steps):
        r_old = r[i - 1]
        v_old = v[i - 1]

        speed_sq = np.dot(v_old, v_old)

        # Helmholtz-Rayleigh active force
        active_force = -beta * (speed_sq - v0**2) * v_old

        # Numerical conservative force: F = -grad U
        grad_U = numerical_grad_U(U, r_old, h=grad_h)
        conservative_force = -grad_U

        # Gaussian noise in 2D
        noise = np.sqrt(2 * D * dt) * rng.normal(size=2)

        # Euler-Maruyama update
        v[i] = v_old + (active_force + conservative_force) * dt + noise
        r[i] = r_old + v_old * dt

        speed[i] = np.linalg.norm(v[i])

        if not np.all(np.isfinite(r[i])) or not np.all(np.isfinite(v[i])):
            print(f"Simulation became unstable at step {i}")
            r = r[:i]
            v = v[:i]
            speed = speed[:i]
            time = time[:i]
            break

    return time, r, v, speed