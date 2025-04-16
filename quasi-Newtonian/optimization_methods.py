import numpy as np
from scipy.optimize import rosen, rosen_der, line_search


class OptimizationMethods:
    @staticmethod
    def newton_cg(f, grad, hess, x0, max_iter=1000, tol=1e-6):
        """Реализация метода Newton-CG"""
        x = x0.copy()
        history = [x.copy()]

        for _ in range(max_iter):
            g = grad(x)
            if np.linalg.norm(g) < tol:
                break

            H = hess(x)
            p = -np.linalg.solve(H, g)

            # Линейный поиск
            alpha = line_search(f, grad, x, p)[0] or 1.0
            x += alpha * p
            history.append(x.copy())

        return x, np.array(history)

    @staticmethod
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

    @staticmethod
    def modified_bfgs(f, grad, x0, max_iter=1000, tol=1e-6):
        """Модифицированный метод BFGS с проверками"""
        n = len(x0)
        I = np.eye(n)
        H = I
        x = x0.copy()
        history = [x.copy()]
        prev_f = f(x)

        for _ in range(max_iter):
            g = grad(x)
            if np.linalg.norm(g) < tol:
                break

            p = -H @ g

            # Линейный поиск с условиями Вольфе
            alpha, x_new = OptimizationMethods.strong_wolfe(f, grad, x, p)
            s = x_new - x
            y = grad(x_new) - g

            # Проверка условия кривизны
            sy = y @ s
            if sy <= 1e-10:
                H = I
                continue

            rho = 1.0 / sy

            # Обновление матрицы H
            A = I - rho * np.outer(s, y)
            B = I - rho * np.outer(y, s)
            H = A @ H @ B + rho * np.outer(s, s)

            # Проверка на уменьшение функции
            curr_f = f(x_new)
            if curr_f > prev_f:
                H = I
            prev_f = curr_f

            x = x_new
            history.append(x.copy())

        return x, np.array(history)

    @staticmethod
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