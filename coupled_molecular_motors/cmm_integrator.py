import numpy as np

def ratchet_force(x, U0=1.0, L=1.0, a=0.3):

    y = np.mod(x, L)

    force = np.where(
        y < a * L,
        -U0 / (a * L),
        U0 / ((1.0 - a) * L)
    )

    return force

def simulate_coupled_molecular_motors(
    num_motors=50,
    dt=1e-3,
    n_steps=100000,
    L=1.0,
    a=0.25,
    U0=5.0,
    q=None,
    d=None,
    r0=1.0,
    r1_hat=15.0,
    k_b=1.0,
    T=0.5,
    fric=5.0,
    F_ext=0.0,
    x0=15.0,
    seed=42
):


    # --------------------------------------------------------
    # Default parameters
    # --------------------------------------------------------

    if q is None:
        q = L / num_motors

    if d is None:
        d = 0.15 * L

    # --------------------------------------------------------
    # RNG
    # --------------------------------------------------------

    rng = np.random.default_rng(seed)

    # --------------------------------------------------------
    # Storage arrays
    # --------------------------------------------------------

    x = np.zeros(n_steps)
    x[0] = x0

    x_j = np.zeros((n_steps, num_motors))

    sigma_history = np.zeros(
        (n_steps, num_motors),
        dtype=int
    )

    # Initial motor states
    sigma = rng.integers(
        0,
        2,
        size=num_motors
    )

    sigma_history[0] = sigma

    # Motor indices
    j_indices = np.arange(num_motors)

    # --------------------------------------------------------
    # Main simulation loop
    # --------------------------------------------------------

    for i in range(1, n_steps):

        # ----------------------------------------------------
        # Shifted motor positions
        # ----------------------------------------------------

        x_j[i] = x[i - 1] + j_indices * q

        # ----------------------------------------------------
        # Position-dependent detachment rates
        # ----------------------------------------------------

        y = np.mod(x_j[i], L)

        near_minimum = (
            (y < d / 2) |
            (y > L - d / 2)
        )

        r1 = np.zeros(num_motors)

        r1[near_minimum] = r1_hat

        # ----------------------------------------------------
        # Transition probabilities
        # ----------------------------------------------------

        p_attach = 1.0 - np.exp(-r0 * dt)

        p_detach = 1.0 - np.exp(-r1 * dt)

        # ----------------------------------------------------
        # Random numbers
        # ----------------------------------------------------

        u = rng.random(num_motors)

        # ----------------------------------------------------
        # Markov updates
        # ----------------------------------------------------

        sigma_new = sigma.copy()

        # Detached -> Attached
        attach_events = (
            (sigma == 0) &
            (u < p_attach)
        )

        # Attached -> Detached
        detach_events = (
            (sigma == 1) &
            (u < p_detach)
        )

        sigma_new[attach_events] = 1

        sigma_new[detach_events] = 0

        # Update sigma
        sigma = sigma_new.copy()

        # Store history
        sigma_history[i] = sigma

        # ----------------------------------------------------
        # Motor force
        # ----------------------------------------------------

        motor_force = np.sum(
            sigma * ratchet_force(
                x_j[i],
                U0=U0,
                L=L,
                a=a
            )
        ) / num_motors

        # ----------------------------------------------------
        # Thermal noise
        # ----------------------------------------------------

        noise = np.sqrt(
            2.0 * k_b * T * dt / fric
        ) * rng.normal()

        # ----------------------------------------------------
        # Langevin update
        # ----------------------------------------------------

        x[i] = (
            x[i - 1]
            + (dt / fric) * (F_ext + motor_force)
            + noise
        )

    # --------------------------------------------------------
    # Derived quantities
    # --------------------------------------------------------

    attached_fraction = np.mean(
        sigma_history,
        axis=1
    )

    time = np.arange(n_steps) * dt

    # --------------------------------------------------------
    # Return results
    # --------------------------------------------------------

    results = {
        "x": x,
        "x_j": x_j,
        "sigma_history": sigma_history,
        "attached_fraction": attached_fraction,
        "time": time,
    }

    return results