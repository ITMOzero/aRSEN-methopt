import typing as tp
from enum import Enum

import numpy as np
import sympy as sp


class GradientDescent:
    class Method(Enum):
        CONSTANT = 1
        DESCENDING = 2
        OPTIMAL = 3
        DICHOTOMY = 4

    def __init__(self, mode: Method) -> None:
        self.mode = mode

    def _get_gradient(self, f: tp.Any, vars: tp.Any) -> tp.Any:
        """
        Вычисляет градиент функции f по переменным variables.
        :param f: исходная функция
        :param vars: список переменных
        :return: функция градиента
        """
        gradient = [sp.diff(f, var) for var in vars]
        return sp.lambdify(vars, gradient, 'numpy')

    def _constant(self, grad: tp.Any, x: np.array, rate: float, iters: int = 1000, eps: float = 1e-6) -> tuple[
        np.array, int]:
        """
        Градиентный спуск с постоянным шагом.
        :param grad: градиент
        :param x: начальная точка
        :param rate: шаг
        :param iters: максимальное число итераций
        :param eps: точность
        :return: точка минимума и число итераций
        """
        i = 0
        x = np.array(x, dtype=float)
        for i in range(iters):
            gradient = np.array(grad(*x))
            tmp = x - rate * gradient
            if np.linalg.norm(tmp - x) < eps:
                break
            x = tmp
        return x, i + 1

    def _descending(self, f: tp.Any, grad: tp.Any, x: np.array, rate: float = 1.0, ratio: float = 0.5,
                    iters: int = 1000,
                    eps: float = 1e-6) -> tuple[np.array, int]:
        """
        Градиентный спуск с дроблением шага.
        :param f: исходная функция
        :param grad: градиент
        :param x: начальная точка
        :param rate: начальный шаг
        :param ratio: коэффициент дробления шага
        :param iters: максимальное число итераций
        :param eps: точность
        :return: точка минимума и число итераций
        """
        i = 0
        x = np.array(x, dtype=float)
        for i in range(iters):
            gradient = np.array(grad(*x))
            while f(*(x - rate * gradient)) > f(*x) - 0.5 * rate * np.linalg.norm(gradient) ** 2:
                rate *= ratio
            tmp = x - rate * gradient
            if np.linalg.norm(tmp - x) < eps:
                break
            x = tmp
        return x, i + 1

    def _optimal(self, f: tp.Any, grad: tp.Any, x: np.array, iters: int = 1000, eps: float = 1e-6) -> tuple[
        np.array, int]:
        """
        Градиентный спуск с поиском оптимального шага.
        :param f: функция
        :param grad: функция градиента
        :param x: начальная точка
        :param iters: максимальное число итераций
        :param eps: точность
        :return: точка минимума и число итераций
        """
        i = 0
        x = np.array(x, dtype=float)
        for i in range(iters):
            gradient = np.array(grad(*x))
            alpha = self._golden_ratio_search(lambda a: f(*(x - a * gradient)), 0, 1)
            tmp = x - alpha * gradient
            if np.linalg.norm(tmp - x) < eps:
                break
            x = tmp
        return x, i + 1

    def _golden_ratio_search(self, f: tp.Any, a: float, b: float, eps: float = 1e-6) -> float:
        """
        Поиск минимума функции f на интервале [a, b] с помощью золотого сечения
        :param f: функция одной переменной.
        :param a: начало интервал
        :param b: конец интервала
        :param eps: точность
        :return: точка минимума
        """
        phi = (np.sqrt(5) - 1) / 2
        c = b - phi * (b - a)
        d = a + phi * (b - a)
        while abs(c - d) > eps:
            if f(c) < f(d):
                b = d
            else:
                a = c
            c = b - phi * (b - a)
            d = a + phi * (b - a)
        return (a + b) / 2

    def _dichotomy(self, f: tp.Any, grad: tp.Any, x: np.array, iters: int = 1000, eps: float = 1e-6) -> tuple[
        np.array, int]:
        """
        Градиентный спуск с поиском оптимального шага по методу дихотомии.
        :param f: функция
        :param grad: функция градиента
        :param x: начальная точка
        :param iters: максимальное число итераций
        :param eps: точность
        :return: точка минимума и число итераций
        """
        i = 0
        x = np.array(x, dtype=float)
        for i in range(iters):
            gradient = np.array(grad(*x))
            alpha = self._dichotomy_search(lambda a: f(*(x - a * gradient)), 0, 1, eps=eps)
            tmp = x - alpha * gradient
            if np.linalg.norm(tmp - x) < eps:
                break
            x = tmp
        return x, i + 1

    def _dichotomy_search(self, f: tp.Any, a: float, b: float, eps: float = 1e-6, delta: float = None) -> float:
        """
        Поиск минимума функции f на интервале [a, b] с помощью метода дихотомии.
        :param f: функция одной переменной.
        :param a: начало интервала
        :param b: конец интервала
        :param eps: требуемая точность
        :param delta: небольшое число для разбиения интервала (если не задано, берётся eps/2)
        :return: приближённое значение аргумента минимума
        """
        if delta is None:
            delta = eps / 2
        while (b - a) > eps:
            mid = (a + b) / 2
            c = mid - delta
            d = mid + delta
            if f(c) < f(d):
                b = d
            else:
                a = c
        return (a + b) / 2


    def find_min(self, f: tp.Any, vars: tp.Any, x: np.array, rate: float = 0.5, ratio: float = 1.0, iters: int = 1000,
                 eps: float = 1e-6) -> \
            tuple[np.array, int]:
        """
        Поиск минимума функции
        :param f: функция
        :param vars: переменные
        :param x: начальная точка
        :param rate: начальный шаг
        :param ratio: коэффициент дробления
        :param iters: максимальное число итераций
        :param eps: точность
        :return:
        """
        grad = self._get_gradient(f, vars)
        f_lambdified = sp.lambdify(vars, f, 'numpy')
        if self.mode == self.Method.CONSTANT:
            return self._constant(grad, x, rate, iters, eps)
        elif self.mode == self.Method.DESCENDING:
            return self._descending(f_lambdified, grad, x, rate, ratio, iters, eps)
        elif self.mode == self.Method.OPTIMAL:
            return self._optimal(f_lambdified, grad, x, iters, eps)
        else:
            raise NotImplementedError


if __name__ == '__main__':
    x, y, z = sp.symbols('x y z')
    f = (x + 124) ** 2 + (y + 410) ** 2 + (z + 4) * (z - 4)

    constant = GradientDescent(GradientDescent.Method.CONSTANT)
    descending = GradientDescent(GradientDescent.Method.DESCENDING)
    optimal = GradientDescent(GradientDescent.Method.OPTIMAL)
    dichotomy = GradientDescent(GradientDescent.Method.DICHOTOMY)

    print('constant rate:')
    print(constant.find_min(f, [x, y, z], [10.0, 10.0, 10.0], iters=10000))

    print('descending rate:')
    print(descending.find_min(f, [x, y, z], [10.0, 10.0, 10.0]))

    print('optimal rate:')
    print(optimal.find_min(f, [x, y, z], [10.0, 10.0, 10.0]))

    print('Optimal rate (dichotomy):')
    print(dichotomy.find_min(f, [x, y, z], [10.0, 10.0, 10.0]))
