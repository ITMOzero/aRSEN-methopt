from backtracking import backtracking
import numpy as np

# Modified BFGS
# Checking the curvature condition y_k^T*s_k>0 before updating H_k.
# Using linear search with strong condition of Wolfe
def bfgs_modified(f, grad, x0, max_iter=1000, tol=1e-6):
    n = len(x0)
    I = np.eye(n)
    H = I  # The initial approximation of the inverse Hessian
    x = x0.copy()
    history = [x.copy()]
    prev_f = f(x)

    for k in range(max_iter):
        g = grad(x)
        if np.linalg.norm(g) < tol:
            break

        p = -H @ g  # direction of search

        alpha, x_new = strong_wolfe(f, grad, x, p)
        s = x_new - x
        y = grad(x_new) - g

        # Checking the curvature condition
        sy = y @ s
        if sy <= 1e-10:
            H = I
            continue

        # Update matrix H with BFGS
        rho = 1.0 / sy
        A = I - rho * np.outer(s, y)
        B = I - rho * np.outer(y, s)
        H = A @ H @ B + rho * np.outer(s, s)

        # Checking for function reduction
        curr_f = f(x_new)
        if curr_f > prev_f + 1e-8:
            H = I

        prev_f = curr_f
        x = x_new
        history.append(x.copy())

    return x, np.array(history)


def strong_wolfe(f, grad, x, p, max_alpha=1.0, c1=1e-4, c2=0.9, max_iter=50):
    alpha = max_alpha
    f0 = f(x)
    g0 = grad(x) @ p

    for _ in range(max_iter):
        x_new = x + alpha * p
        f_alpha = f(x_new)
        g_alpha = grad(x_new) @ p

        # Checking for sufficient decreasing (Armijo)
        if f_alpha > f0 + c1 * alpha * g0:
            alpha *= 0.5
            continue

        # Checking the curvature (strong Wolfe condition)
        if abs(g_alpha) > -c2 * g0:
            alpha *= 1.5
        else:
            break

    return alpha, x_new


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
