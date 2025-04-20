from backtracking import backtracking
import numpy as np

# TODO repair
def dfp_method(f: callable, grad: callable, hess: callable = None,
               x_0: float = 0.0, eps: float = 1e-6, max_iter: int = 100,
               params: dict = None):
    x = np.array([x_0])
    path = [x[0]]
    H = np.eye(1)
    g = grad(x)

    i = 0
    while i <= max_iter:
        if np.linalg.norm(g) < eps:
            break

        p = -np.dot(H, g)
        alpha = backtracking(f, x, f(x), g, p, **(params or {}))
        x_new = x + alpha * p
        g_new = grad(x_new)

        s, y = x_new - x, g_new - g

        ys = np.dot(y, s)
        if ys > 1e-10 and np.all(np.isfinite(s)) and np.all(np.isfinite(y)):
            rho = 1.0 / ys
            Hy = np.dot(H, y)
            H += rho * (np.outer(s, s) - np.outer(Hy, Hy))

        x, g = x_new, g_new
        path.append(x[0])
        i += 1

    return x[0], i, path