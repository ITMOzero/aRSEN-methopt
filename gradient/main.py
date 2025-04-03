import typing as tp
from enum import Enum

import numpy as np
import sympy as sp

from functions import *
from graphics import animate_2d
from graphics import plot_3d


class GradientDescent:
    class Method(Enum):
        CONSTANT = 1
        DESCENDING = 2
        GOLDEN_RATIO = 3
        DICHOTOMY = 4

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

    def _constant(self, grad: tp.Any, point: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Градиентный спуск с постоянным шагом.
        Метод использует фиксированный шаг для перемещения по направлению градиента на каждой итерации.
        :param grad: градиент функции
        :param point: начальная точка
        :param learning_rate: шаг обучения
        :return: новая точка, перемещенная по направлению градиента
        """
        grad0 = np.array(grad(*point))
        gradient = grad0 / np.linalg.norm(grad0)
        tmp = point - learning_rate * gradient

        return tmp

    def _descending(self, grad: tp.Any, point: np.ndarray, iter: int, initial_learning_rate: float = 3.0, max_ratio: int = 15,
                    ) -> np.ndarray:
        """
        Градиентный спуск с уменьшением шага.
        Метод уменьшает шаг на каждой итерации для более точного подхода к минимуму.
        :param grad: градиент функции
        :param point: начальная точка
        :param initial_learning_rate: начальный шаг
        :param max_ratio: максимальный коэффициент уменьшения шага
        :param iter: номер текущей итерации
        :return: новая точка, перемещенная по направлению градиента с уменьшенным шагом
        """
        initial_learning_rate = 5
        grad0 = np.array(grad(*point))
        gradient = grad0 / np.linalg.norm(grad0)
        learning_rate = initial_learning_rate / min(max_ratio, iter + 1)
        tmp = point - learning_rate * gradient

        return tmp

    def _golden_ratio(self, f: tp.Any, grad: tp.Any, point: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """
        Градиентный спуск с поиском оптимального шага методом золотого сечения.
        Метод использует золотое сечение для нахождения оптимального шага в направлении градиента.
        :param f: исходная функция
        :param grad: градиент функции
        :param point: начальная точка
        :param eps: точность
        :return: новая точка, перемещенная по направлению градиента с найденным оптимальным шагом
        """
        grad0 = np.array(grad(*point))
        gradient = grad0 / np.linalg.norm(grad0)
        a = 0
        b = 1

        phi = (np.sqrt(5) - 1) / 2
        c = b - phi * (b - a)
        d = a + phi * (b - a)
        while abs(c - d) > eps:
            if f(*(point - c * gradient)) < f(*(point - d * gradient)):
                b = d
            else:
                a = c
            c = b - phi * (b - a)
            d = a + phi * (b - a)

        learning_rate = (a + b) / 2
        tmp = point - learning_rate * gradient
        return tmp

    def _dichotomy(self, f: tp.Any, grad: tp.Any, point: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """
        Градиентный спуск с поиском оптимального шага методом дихотомии.
        Метод делит интервал для шага пополам и выбирает наименьший шаг для продвижения в сторону минимума.
        :param f: исходная функция
        :param grad: градиент функции
        :param point: начальная точка
        :param eps: точность
        :return: новая точка, перемещенная по направлению градиента с найденным оптимальным шагом
        """
        grad0 = np.array(grad(*point))
        gradient = grad0 / np.linalg.norm(grad0)
        a = 0
        b = 1
        while (b - a) > eps:
            c = (a + b) / 2
            left = (a + c) / 2
            right = (c + b) / 2

            if f(*(point - left * gradient)) < f(*(point - c * gradient)):
                b = c
            elif f(*(point - right * gradient)) < f(*(point - c * gradient)):
                a = c
            else:
                a = left
                b = right

        learning_rate = (a + b) / 2
        tmp = point - learning_rate * gradient

        return tmp

    def _arrays_equals(self, a: np.ndarray, b: np.ndarray) -> bool:
        """
        Проверка равенства двух массивов.
        :param a: первый массив
        :param b: второй массив
        :return: True, если массивы одинаковы, иначе False
        """
        if a.size != b.size:
            return False
        for i in range(a.size):
            if a[i] != b[i]:
                return False
        return True

    import numpy as np

    def find_min(self, f: tp.Any, variables: tp.Any, starting_point: np.ndarray, learning_rate: float = 0.5,
                 max_ratio: int = 20, iters: int = 50,
                 eps: float = 1e-6) -> \
            tuple[np.ndarray, int, np.ndarray]:
        """
        Поиск минимума функции с использованием градиентного спуска.
        :param f: целевая функция
        :param variables: переменные, по которым вычисляется градиент
        :param starting_point: начальная точка для поиска минимума
        :param learning_rate: начальный шаг
        :param max_ratio: максимальный коэффициент дробления шага
        :param iters: максимальное количество итераций
        :param eps: точность, с которой нужно найти минимум
        :return: точка минимума, количество итераций и траектория поиска
        """

        point = np.array(starting_point, dtype=float)
        grad = self._get_gradient(f, variables)
        f_lambdified = sp.lambdify(variables, f, 'numpy')

        i = 0
        trajectory = [point.copy()]
        prev: np.ndarray = -1 * point
        for i in range(iters):

            if self.mode == self.Method.CONSTANT:
                tmp = self._constant(grad, point, learning_rate)
            elif self.mode == self.Method.DESCENDING:
                tmp = self._descending(grad, point,i, learning_rate, max_ratio)
            elif self.mode == self.Method.GOLDEN_RATIO:
                tmp = self._golden_ratio(f_lambdified, grad, point, eps)
            elif self.mode == self.Method.DICHOTOMY:
                tmp = self._dichotomy(f_lambdified, grad, point, eps)
            else:
                raise NotImplementedError

            trajectory.append(tmp.copy())
            if np.linalg.norm(tmp - point) < eps or self._arrays_equals(tmp, prev):
                break
            prev = point
            point = tmp

        return point, i + 1, np.array(trajectory)



def animate(variables, f, label, method):
    """
    Анимация траектории градиентного спуска для заданной функции.
    :param variables: переменные функции
    :param f: функция для минимизации
    :param label: метка для визуализации
    :param method: метод градиентного спуска
    """

    print(f'==== {label} ====')
    res, iterations, trajectory = method.find_min(f, variables, np.array([-10.0]), learning_rate=0.2)
    print(f'result: {res}, in {iterations} steps')
    print('close plot window to continue')
    animate_2d(sp.lambdify([x], f, 'numpy'), trajectory, variables, xlim)


if __name__ == '__main__':
    x, y, z = sp.symbols('x y z')
    f, variables, starting_point = select_function('scipy_f1')

    constant = GradientDescent(GradientDescent.Method.CONSTANT)
    descending = GradientDescent(GradientDescent.Method.DESCENDING)
    optimal = GradientDescent(GradientDescent.Method.GOLDEN_RATIO)
    dichotomy = GradientDescent(GradientDescent.Method.DICHOTOMY)

    trajectories = []
    labels = []
    for method_name, method in [
        ('Constant learning_rate', constant),
        ('Descending learning_rate', descending),
        ('Optimal learning_rate (gold ratio)', optimal),
        ('Optimal learning_rate (dichotomy)', dichotomy),
    ]:
        print(f'{method_name}:')
        point, step, trajectory = method.find_min(f, variables, starting_point)
        labels.append(method_name.replace(' ', '_'))
        trajectories.append(trajectory)
        point_formatted = [{str(var): f'{p:.16f}'} for var, p in zip([x, y, z], point)]
        print(point_formatted, step, "\n")

    xlim = (-10, 10)
    ylim = (-10, 10)

    plot_3d(sp.lambdify(variables, f, 'numpy'), trajectories, labels, variables, xlim, ylim)

    g = x ** 2 + x / 3 - 5
    animate([x], g, 'constant', constant)
    animate([x], g, 'descending', descending)
    animate([x], g, 'golden ration', optimal)
    animate([x], g, 'dichotomy', dichotomy)

