import math
import typing as tp
from enum import Enum

import numpy as np
import sympy as sp
from matplotlib import pyplot as plt


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
        np.array, int, np.array]:
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
        trajectory = [x.copy()]
        for i in range(iters):
            gradient = np.array(grad(*x))
            tmp = x - rate * gradient
            trajectory.append(tmp.copy())
            if np.linalg.norm(tmp - x) < eps:
                break
            x = tmp
        return x, i + 1, np.array(trajectory)

    def _descending(self, f: tp.Any, grad: tp.Any, x: np.array, rate: float = 1.0, ratio: float = 0.5,
                    iters: int = 1000, eps: float = 1e-6) -> tuple[np.array, int, np.array]:
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
        trajectory = [x.copy()]
        for i in range(iters):
            gradient = np.array(grad(*x))

            norm_gradient = np.linalg.norm(gradient)
            if norm_gradient != 0:
                gradient = gradient / norm_gradient

            while f(*(x - rate * gradient)) > f(*x) - 0.5 * rate * np.linalg.norm(gradient) ** 2:
                rate *= ratio
            tmp = x - rate * gradient
            trajectory.append(tmp.copy())
            if np.linalg.norm(tmp - x) < eps:
                break
            x = tmp

            rate = rate / (1 + i)

        return x, i + 1, np.array(trajectory)

    def _optimal(self, f: tp.Any, grad: tp.Any, x: np.array, iters: int = 1000, eps: float = 1e-6) -> tuple[
        np.array, int, np.array]:
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
        trajectory = [x.copy()]
        for i in range(iters):
            gradient = np.array(grad(*x))
            alpha = self._golden_ratio_search(lambda a: f(*(x - a * gradient)), 0, 1)
            tmp = x - alpha * gradient
            trajectory.append(tmp.copy())
            if np.linalg.norm(tmp - x) < eps:
                break
            x = tmp
        return x, i + 1, np.array(trajectory)

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
        np.array, int,np.array]:
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
        trajectory = [x.copy()]
        for i in range(iters):
            gradient = np.array(grad(*x))
            alpha = self._dichotomy_search(lambda a: f(*(x - a * gradient)), 0, 1, eps=eps)
            tmp = x - alpha * gradient
            trajectory.append(tmp.copy())
            if np.linalg.norm(tmp - x) < eps:
                break
            x = tmp
        return x, i + 1, np.array(trajectory)

    def _dichotomy_search(self, f: tp.Any, a: float, b: float, eps: float = 1e-6) -> float:
        """
        Поиск минимума функции f на интервале [a, b] с помощью метода дихотомии.
        :param f: функция одной переменной.
        :param a: начало интервала
        :param b: конец интервала
        :param eps: требуемая точность
        :return: приближённое значение аргумента минимума
        """
        while (b - a) > eps:
            c = (a + b) / 2
            left = (a + c) / 2
            right = (c + b) / 2

            if f(left) < f(c):
                b = c
            elif f(right) < f(c):
                a = c
            else:
                a = left
                b = right

        return (a + b) / 2

    def find_min(self, f: tp.Any, vars: tp.Any, x: np.array, rate: float = 0.5, ratio: float = 1.0, iters: int = 1000,
                 eps: float = 1e-6) -> \
            tuple[np.array, int, np.array]:
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
        elif self.mode == self.Method.DICHOTOMY:
            return self._dichotomy(f_lambdified, grad, x, iters, eps)
        else:
            raise NotImplementedError



def plot_3d(f, trajectory_constant, trajectory_descending, trajectory_optimal, trajectory_dichotomy, vars, xlim, ylim):
    plt.figure(figsize=(10, 8))


    ax = plt.axes(projection='3d')



    # ax.plot(trajectory_constant[:, 0], trajectory_constant[:, 1], f(trajectory_constant[:, 0], trajectory_constant[:, 1]), 'r-', label='Constant rate')
    ax.plot(trajectory_descending[:, 0], trajectory_descending[:, 1], f(trajectory_descending[:, 0], trajectory_descending[:, 1]), 'g-', label='Descending rate')
    ax.plot(trajectory_optimal[:, 0], trajectory_optimal[:, 1], f(trajectory_optimal[:, 0], trajectory_optimal[:, 1]), 'b-', label='Optimal rate (gold ratio)')
    ax.plot(trajectory_dichotomy[:, 0], trajectory_dichotomy[:, 1], f(trajectory_dichotomy[:, 0], trajectory_dichotomy[:, 1]), 'y-', label='Optimal rate (dichotomy)')

    ax.set_xlabel(vars[0])
    ax.set_ylabel(vars[1])
    ax.set_zlabel('f(x, y)')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.legend()

    plt.savefig('gradient_descent_3d_plot.png')




if __name__ == '__main__':
    # x, y, z = sp.symbols('x y z')
    # f = (x ** 2) * 2 + 3 * (y ** 2) + (z ** 2)/2
    #
    # constant = GradientDescent(GradientDescent.Method.CONSTANT)
    # descending = GradientDescent(GradientDescent.Method.DESCENDING)
    # optimal = GradientDescent(GradientDescent.Method.OPTIMAL)
    # dichotomy = GradientDescent(GradientDescent.Method.DICHOTOMY)
    #
    #
    # # TODO: is it correct?
    # print('Constant rate:')
    # print(constant.find_min(f, [x, y, z], [10.0, 10.0, 10.0]))
    #
    # print('Descending rate:')
    # print(descending.find_min(f, [x, y, z], [10.0, 10.0, 10.0]))
    #
    # print('Optimal rate (gold ratio):')
    # print(optimal.find_min(f, [x, y, z], [10.0, 10.0, 10.0]))
    #
    # print('Optimal rate (dichotomy):')
    # print(dichotomy.find_min(f, [x, y, z], [10.0, 10.0, 10.0]))

    x, y = sp.symbols('x y')
    f = (x ** 2) * 2 + 3 * (y ** 2)  # Example function

    constant = GradientDescent(GradientDescent.Method.CONSTANT)
    descending = GradientDescent(GradientDescent.Method.DESCENDING)
    optimal = GradientDescent(GradientDescent.Method.OPTIMAL)
    dichotomy = GradientDescent(GradientDescent.Method.DICHOTOMY)


    _, _, trajectory_constant = constant.find_min(f, [x, y], [10.0, 10.0])
    _, _, trajectory_descending = descending.find_min(f, [x, y], [10.0, 10.0])
    _, _, trajectory_optimal = optimal.find_min(f, [x, y], [10.0, 10.0])
    _, _, trajectory_dichotomy = dichotomy.find_min(f, [x, y], [10.0, 10.0])

    xlim = (-10, 10)
    ylim = (-10, 10)

    plot_3d(sp.lambdify([x, y], f, 'numpy'), trajectory_constant, trajectory_descending, trajectory_optimal,
            trajectory_dichotomy, [x, y], xlim, ylim)

