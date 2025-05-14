import scipy.optimize as opt
import numpy as np

def scipy_method(f: callable, grad: callable, hess: callable=None,
                 x_0: float=None, method: str= 'BFGS', options: dict=None):
    path = []
    callback = lambda x_k: path.append(float(x_k[0]) if isinstance(x_k, np.ndarray) else float(x_k))
    res = opt.minimize(f, np.array(x_0), method=method, jac=grad, hess=hess,
                       callback=callback, options=options)
    return res.x[0], res.nit, path


# квазиньютоновский метод для ОСН ЗАДАНИЯ 2
def bfgs(f, grad, x0, max_iter=1000, tol=1e-6):
    """Реализация метода BFGS"""
    n = len(x0)
    I = np.eye(n)
    H = I
    x = x0.copy()
    history = [x.copy()]

    for _ in range(max_iter):
        g = grad(x)
        if np.linalg.norm(g) < tol:
            break

        p = -H @ g

        # Линейный поиск
        alpha = line_search(f, grad, x, p)[0] or 1.0
        x_new = x + alpha * p
        s = x_new - x
        y = grad(x_new) - g

        # Обновление матрицы H
        rho = 1.0 / (y @ s)
        A = I - rho * np.outer(s, y)
        B = I - rho * np.outer(y, s)
        H = A @ H @ B + rho * np.outer(s, s)

        x = x_new
        history.append(x.copy())

    return x, np.array(history)

def line_search(f, grad, x, p, max_alpha=1.0, c1=1e-4, c2=0.9):
    """Упрощенный линейный поиск с условиями Вольфе"""
    alpha = max_alpha
    f0 = f(x)
    g0 = grad(x) @ p

    while True:
        x_new = x + alpha * p
        f_alpha = f(x_new)
        if f_alpha > f0 + c1 * alpha * g0:
            alpha *= 0.5
        else:
            g_alpha = grad(x_new) @ p
            if g_alpha < c2 * g0:
                alpha *= 1.5
            else:
                break
    return alpha