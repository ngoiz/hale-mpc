import numpy as np


def force_exp(t_dom, alpha, out_value):
    """
    Reference:

    https://caefn.com/cfd/hyperbolic-tangent-stretching-grid
    """
    t = t_dom - t_dom[0]
    zeta = -1 + 2 * t/t[-1]
    y_out = 1 / alpha * np.tanh(zeta * np.arctanh(alpha)) + 1

    return y_out * out_value * 0.5


def generate_force_array(n_tsteps, dt, alpha, force, t_init, t_end):
    t_dom = np.linspace(0, n_tsteps * dt, n_tsteps)

    idx_init = np.where(t_dom >= t_init)[0][0]
    idx_end = np.where(t_dom >= t_end)[0][0]

    y_out = np.zeros_like(t_dom)
    y_out[idx_init:idx_end] = force_exp(t_dom[idx_init:idx_end], alpha, force)

    y_out[idx_end:] = force

    return t_dom, y_out



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    n_tsteps = 100
    dt = 0.1

    force = 10
    t_init = 2
    t_end = 8
    for alpha in np.linspace(0.9, 0.99, 5):
        plt.plot(*generate_force_array(n_tsteps, dt, alpha, force, t_init, t_end))
    plt.grid()
    plt.show()