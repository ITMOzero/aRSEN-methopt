import numpy as np
from scipy.optimize import minimize
from scipy.optimize import rosen, rosen_der, rosen_hess

# Целевая функция (функция Розенброка)
def objective(x):
    return rosen(x)

# Градиент функции
def gradient(x):
    return rosen_der(x)

# Гессиан функции
def hessian(x):
    return rosen_hess(x)

# Начальная точка
x0 = np.array([-1.2, 1.0])

# Метод Newton-CG
res_newton_cg = minimize(objective, x0, method='Newton-CG',
                        jac=gradient, hess=hessian)
print("Newton-CG result:", res_newton_cg.x)

# Квазиньютоновские методы: BFGS и L-BFGS-B
res_bfgs = minimize(objective, x0, method='BFGS', jac=gradient)
print("BFGS result:", res_bfgs.x)

res_lbfgs = minimize(objective, x0, method='L-BFGS-B', jac=gradient)
print("L-BFGS-B result:", res_lbfgs.x)


# =============================================================


def bfgs_optimize(f, grad, x0, max_iter=1000, tol=1e-6):
    n = len(x0)
    I = np.eye(n)
    H_k = I  # Начальное приближение обратного гессиана
    x_k = x0.copy()

    for k in range(max_iter):
        g_k = grad(x_k)
        if np.linalg.norm(g_k) < tol:
            break

        p_k = -H_k @ g_k  # Направление поиска

        # Линейный поиск (упрощенный)
        alpha = line_search(f, grad, x_k, p_k)

        x_k1 = x_k + alpha * p_k
        s_k = x_k1 - x_k
        y_k = grad(x_k1) - g_k

        # Обновление матрицы H по формуле BFGS
        rho_k = 1.0 / (y_k @ s_k)
        A = I - rho_k * np.outer(s_k, y_k)
        B = I - rho_k * np.outer(y_k, s_k)
        H_k = A @ H_k @ B + rho_k * np.outer(s_k, s_k)

        x_k = x_k1

    return x_k


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


# Тестирование собственной реализации
x_opt = bfgs_optimize(rosen, rosen_der, x0)
print("Custom BFGS result:", x_opt)


def modified_bfgs(f, grad, x0, max_iter=1000, tol=1e-6, reset_thresh=1e6):
    n = len(x0)
    I = np.eye(n)
    H_k = I
    x_k = x0.copy()
    prev_f = f(x_k)

    for k in range(max_iter):
        g_k = grad(x_k)
        if np.linalg.norm(g_k) < tol:
            break

        p_k = -H_k @ g_k

        # Адаптивный линейный поиск
        alpha, x_k1 = strong_wolfe(f, grad, x_k, p_k)

        s_k = x_k1 - x_k
        y_k = grad(x_k1) - g_k

        # Проверка условия кривизны
        sy = y_k @ s_k
        if sy <= 0:
            # Сброс к единичной матрице при нарушении условия
            H_k = I
            continue

        rho_k = 1.0 / sy

        # Проверка на численную стабильность
        if rho_k > reset_thresh:
            H_k = I
            continue

        A = I - rho_k * np.outer(s_k, y_k)
        B = I - rho_k * np.outer(y_k, s_k)
        H_k = A @ H_k @ B + rho_k * np.outer(s_k, s_k)

        # Проверка на уменьшение функции
        curr_f = f(x_k1)
        if curr_f > prev_f:
            H_k = I  # Сброс если функция увеличилась
        prev_f = curr_f

        x_k = x_k1

    return x_k


def strong_wolfe(f, grad, x, p, max_alpha=1.0, c1=1e-4, c2=0.9, max_iter=20):
    """Линейный поиск с сильными условиями Вольфе"""
    alpha = max_alpha
    f0 = f(x)
    g0 = grad(x) @ p

    for _ in range(max_iter):
        x_new = x + alpha * p
        f_alpha = f(x_new)
        g_alpha = grad(x_new) @ p

        if f_alpha > f0 + c1 * alpha * g0 or (f_alpha >= f(x + alpha / 2 * p) and _ > 0):
            alpha *= 0.5
        elif abs(g_alpha) <= -c2 * g0:
            break
        else:
            alpha *= 1.5

    return alpha, x + alpha * p




