import numpy as np

def green_function_damped(E, X, Y, Z, D, t, tau, Tc):

    if t <= 0:
        return E + Tc

    nx, ny, nz = E.shape

    r = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    E0 = E.ravel()

    U = np.zeros_like(E0)

    norm = (4 * np.pi * D * t) ** 1.5
    decay = np.exp(-t / tau)

    for i in range(len(E0)):

        Ei = E0[i]
        if Ei == 0:
            continue

        ri = r[i]

        dr = r - ri
        dr2 = np.sum(dr * dr, axis=1)

        U += Ei * np.exp(-dr2 / (4 * D * t)) * decay / norm

    return Tc + U.reshape(nx, ny, nz)

