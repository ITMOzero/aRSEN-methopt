import typing as tp
from enum import Enum

import numpy as np
import sympy as sp
from graphics import plot_3d

class GradientDescent:
    class Method(Enum):
        CONSTANT = 1
        DESCENDING = 2
        GOLDEN_RATIO = 3
        DICHOTOMY = 4
        ADAPTIVE = 5

    def __init__(self, mode: Method) -> None:
        self.mode = mode

    def _get_gradient(self, f: tp.Any, variables: tp.Any) -> tp.Any:
        """
        Вычисляет градиент функции f по переменным variables.
        :param f: исходная функция
        :param variables: список переменных
        :return: функция градиента
        """
        gradient = [sp.diff(f, var) for var in variables]
        return sp.lambdify(variables, gradient, 'numpy')

    def _constant(self, grad: tp.Any, point: np.ndarray, learning_rate: float, iters: int = 1000, eps: float = 1e-6) -> tuple[
        np.ndarray, int, np.ndarray]:
        """
        Градиентный спуск с постоянным шагом.
        :param grad: градиент
        :param point: начальная точка
        :param learning_rate: шаг
        :param iters: максимальное число итераций
        :param eps: точность
        :return: точка минимума и число итераций
        """
        i = 0
        trajectory = [point.copy()]
        for i in range(iters):

            gradient = np.array(grad(*point))
            tmp = point - learning_rate * gradient

            trajectory.append(tmp.copy())
            if np.linalg.norm(tmp - point) < eps:
                break
            point = tmp
        return point, i + 1, np.array(trajectory)

    def _descending(self, f: tp.Any, grad: tp.Any, point: np.ndarray, learning_rate: float = 1.0, ratio: float = 0.5,
                    iters: int = 1000, eps: float = 1e-6) -> tuple[np.ndarray, int, np.ndarray]:
        """
        Градиентный спуск с дроблением шага.
        :param f: исходная функция
        :param grad: градиент
        :param point: начальная точка
        :param learning_rate: начальный шаг
        :param ratio: коэффициент дробления шага
        :param iters: максимальное число итераций
        :param eps: точность
        :return: точка минимума и число итераций
        """
        i = 0
        trajectory = [point.copy()]
        initial_learning_rate = learning_rate

        for i in range(iters):
            gradient = np.array(grad(*point))
            learning_rate = initial_learning_rate

            while f(*(point - learning_rate * gradient)) > f(*point) - 0.5 * learning_rate * np.linalg.norm(gradient) ** 2:
                learning_rate *= ratio
                if learning_rate < 1e-10:
                    break

            tmp = point - learning_rate * gradient
            trajectory.append(tmp.copy())

            if np.linalg.norm(tmp - point) < eps:
                break
            point = tmp

        return point, i + 1, np.array(trajectory)

    def _golden_ratio(self, f: tp.Any, grad: tp.Any, point: np.ndarray, iters: int = 1000, eps: float = 1e-6) -> tuple[
        np.ndarray, int, np.ndarray]:
        """
        Градиентный спуск с поиском оптимального шага.
        :param f: функция
        :param grad: функция градиента
        :param point: начальная точка
        :param iters: максимальное число итераций
        :param eps: точность
        :return: точка минимума и число итераций
        """
        i = 0
        trajectory = [point.copy()]
        for i in range(iters):
            gradient = np.array(grad(*point))
            learning_rate = self._golden_ratio_search(lambda a: f(*(point - a * gradient)), 0, 1)
            tmp = point - learning_rate * gradient
            trajectory.append(tmp.copy())
            if np.linalg.norm(tmp - point) < eps:
                break
            point = tmp
        return point, i + 1, np.array(trajectory)

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

    def _dichotomy(self, f: tp.Any, grad: tp.Any, point: np.ndarray, iters: int = 1000, eps: float = 1e-6) -> tuple[
        np.ndarray, int, np.ndarray]:
        """
        Градиентный спуск с поиском оптимального шага по методу дихотомии.
        :param f: функция
        :param grad: функция градиента
        :param point: начальная точка
        :param iters: максимальное число итераций
        :param eps: точность
        :return: точка минимума и число итераций
        """
        i = 0
        trajectory = [point.copy()]
        for i in range(iters):
            gradient = np.array(grad(*point))
            learning_rate = self._dichotomy_search(lambda a: f(*(point - a * gradient)), 0, 1)
            tmp = point - learning_rate * gradient
            trajectory.append(tmp.copy())
            if np.linalg.norm(tmp - point) < eps:
                break
            point = tmp
        return point, i + 1, np.array(trajectory)

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

    def _adaptive(self, grad: tp.Any, point: np.ndarray, learning_rate: float, iters: int = 1000, eps: float = 1e-6) -> tuple[
        np.ndarray, int, np.ndarray]:
        """
        Градиентный спуск с адаптивным шагом.
        :param grad: градиент
        :param point: начальная точка
        :param learning_rate: начальный шаг
        :param iters: максимальное число итераций
        :param eps: точность
        :return: точка минимума и число итераций
        """
        i = 0
        trajectory = [point.copy()]
        prev_step = 0
        for i in range(iters):
            gradient = np.array(grad(*point))

            step = learning_rate / (1 + prev_step)
            prev_step = np.linalg.norm(gradient)

            tmp = point - step * gradient
            trajectory.append(tmp.copy())

            if np.linalg.norm(tmp - point) < eps:
                break
            point = tmp
        return point, i + 1, np.array(trajectory)



    def find_min(self, f: tp.Any, variables: tp.Any, starting_point: np.ndarray, learning_rate: float = 0.1, ratio: float = 1.0, iters: int = 1000,
                 eps: float = 1e-6) -> \
            tuple[np.ndarray, int, np.ndarray]:
        """
        Поиск минимума функции
        :param f: функция
        :param variables: переменные
        :param starting_point: начальная точка
        :param learning_rate: начальный шаг
        :param ratio: коэффициент дробления
        :param iters: максимальное число итераций
        :param eps: точность
        :return:
        """

        starting_point = np.array(starting_point, dtype=float)
        grad = self._get_gradient(f, variables)
        f_lambdified = sp.lambdify(variables, f, 'numpy')

        if self.mode == self.Method.CONSTANT:
            return self._constant(grad, starting_point, learning_rate, iters, eps)
        elif self.mode == self.Method.DESCENDING:
            return self._descending(f_lambdified, grad, starting_point, learning_rate, ratio, iters, eps)
        elif self.mode == self.Method.OPTIMAL:
            return self._optimal(f_lambdified, grad, starting_point, iters, eps)
        elif self.mode == self.Method.DICHOTOMY:
            return self._dichotomy(f_lambdified, grad, starting_point, iters, eps)
        elif self.mode == self.Method.ADAPTIVE:
            return self._adaptive(grad, starting_point, learning_rate, iters, eps)
        else:
            raise NotImplementedError


if __name__ == '__main__':
    x, y, z = sp.symbols('x y z')
    # f = (x ** 2) * 2 + 3 * (y ** 2) + (z ** 2)/2

    # f = (x ** 2) * 2 + 3 * (y ** 2)
    f = y ** 2 + x ** 2 / 3

    constant = GradientDescent(GradientDescent.Method.CONSTANT)
    descending = GradientDescent(GradientDescent.Method.DESCENDING)
    optimal = GradientDescent(GradientDescent.Method.GOLDEN_RATIO)
    dichotomy = GradientDescent(GradientDescent.Method.DICHOTOMY)
    adaptive = GradientDescent(GradientDescent.Method.ADAPTIVE)

    trajectories = []
    labels = []
    for method_name, method in [
        ('Constant learning_rate', constant),
        ('Descending learning_rate', descending),
        ('Optimal learning_rate (gold ratio)', optimal),
        ('Optimal learning_rate (dichotomy)', dichotomy),
        ('Adaptive learning_rate', adaptive)
    ]:
        print(f'{method_name}:')
        point, step, trajectory = method.find_min(f, [x, y], np.array([10.0, 10.0]))
        labels.append(method_name.replace(' ', '_'))
        trajectories.append(trajectory)
        point_formatted = [{str(var): f'{p:.16f}'} for var, p in zip([x, y, z], point)]
        print(point_formatted, step, "\n")

    xlim = (-10, 10)
    ylim = (-10, 10)

    plot_3d(sp.lambdify([x, y], f, 'numpy'), trajectories, labels, [x, y], xlim, ylim)


