import typing as tp
from enum import Enum

import numpy as np
from scipy.optimize import *
from functions import *
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


    def _descending(self, f: tp.Any, grad: tp.Any, point: np.ndarray, learning_rate: float = 1.0, ratio: float = 0.5) -> np.ndarray:
        """
        Градиентный спуск с уменьшением шага по мере прогресса.
        Метод итеративно уменьшает шаг (learning_rate), если функция в новой точке не уменьшается
        достаточно сильно.
        :param f: исходная функция
        :param grad: градиент
        :param point: начальная точка
        :param learning_rate: начальный шаг
        :param ratio: коэффициент дробления шага
        :return: новая точка, перемещенная по направлению градиента
        """

        gradient = np.array(grad(*point))
        while f(*(point - learning_rate * gradient)) > f(*point) - 0.5 * learning_rate * np.linalg.norm(gradient) ** 2:
            learning_rate *= ratio
            if learning_rate < 1e-10:
                break
        tmp = point - learning_rate * gradient

        return tmp
    def _constant(self, grad: tp.Any, point: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Градиентный спуск с постоянным шагом.
        Метод использует фиксированный шаг для перемещения по направлению градиента на каждой итерации.
        :param grad: градиент
        :param point: начальная точка
        :param learning_rate: шаг
        :return: новая точка, перемещенная по направлению градиента
        """
        grad0 = np.array(grad(*point))
        gradient = grad0 / np.linalg.norm(grad0)
        tmp = point - learning_rate * gradient

        return tmp



    def _golden_ratio(self, f: tp.Any, grad: tp.Any, point: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """
        Градиентный спуск с поиском оптимального шага методом золотого сечения.
        Метод использует золотое сечение для нахождения оптимального шага в направлении градиента.
        :param f: исходная функция
        :param grad: функция градиента
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



    def _dichotomy(self, f: tp.Any, grad: tp.Any, point: np.ndarray, eps: float = 1e-6) -> tuple[
        np.ndarray, int, np.ndarray]:
        """
        Градиентный спуск с поиском оптимального шага методом дихотомии.
        Метод делит интервал для шага пополам и выбирает наименьший шаг для продвижения в сторону минимума.
        :param f: исходная функция
        :param grad: градиент
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



    def _adaptive(self, grad: tp.Any, point: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Градиентный спуск с адаптивным шагом.
        Метод адаптирует шаг в зависимости от величины градиента.
        :param grad: градиент
        :param point: начальная точка
        :param learning_rate: начальный шаг
        :return: точка минимума и число итераций
        """

        gradient = np.array(grad(*point))
        step = learning_rate / (1 + gradient)
        tmp = point - step * gradient
        return tmp




    def find_min(self, f: tp.Any, variables: tp.Any, point: np.ndarray, learning_rate: float = 0.5, ratio: float = 0.5, iters: int = 1000,
                 eps: float = 1e-6) -> \
            tuple[np.ndarray, int, np.ndarray]:
        """
        Поиск минимума функции
        :param f: функция
        :param variables: переменные
        :param point: начальная точка
        :param learning_rate: начальный шаг
        :param ratio: коэффициент дробления
        :param iters: максимальное число итераций
        :param eps: точность
        :return:
        """

        point = np.array(point, dtype=float)
        grad = self._get_gradient(f, variables)
        f_lambdified = sp.lambdify(variables, f, 'numpy')
        i = 0
        trajectory = [point.copy()]
        for i in range(iters):

            if self.mode == self.Method.CONSTANT:
                tmp = self._constant(grad, point, learning_rate)
            elif self.mode == self.Method.DESCENDING:
                tmp = self._descending(f_lambdified, grad, point, learning_rate, ratio)
            elif self.mode == self.Method.GOLDEN_RATIO:
                tmp = self._golden_ratio(f_lambdified, grad, point, eps)
            elif self.mode == self.Method.DICHOTOMY:
                tmp = self._dichotomy(f_lambdified, grad, point, eps)
            elif self.mode == self.Method.ADAPTIVE:
                tmp = self._adaptive(grad, point, learning_rate)
            else:
                raise NotImplementedError

            trajectory.append(tmp.copy())
            if np.linalg.norm(tmp - point) < eps:
                break
            point = tmp

        return point, i + 1, np.array(trajectory)






def animate(variables, f, label, method):
    print(f'==== {label} ====')
    res, iterations, trajectory = method.find_min(f, variables, np.array([-10.0]), learning_rate=0.2)
    print(f'result: {res}, in {iterations} steps')
    print('close plot window to continue')
    # animate_2d(sp.lambdify([x], f, 'numpy'), trajectory, variables, xlim)



if __name__ == '__main__':
    x, y, z = sp.symbols('x y z')
    # f = (x ** 2) * 2 + 3 * (y ** 2) + (z ** 2)/2

    # f = (x ** 2) * 2 + 3 * (y ** 2)
    # f = y ** 2 * 2+ x ** 2 / 3 + 5
    f, variables, starting_point = select_function('f2')

    constant = GradientDescent(GradientDescent.Method.CONSTANT)
    descending = GradientDescent(GradientDescent.Method.DESCENDING)
    golden_ratio = GradientDescent(GradientDescent.Method.GOLDEN_RATIO)
    dichotomy = GradientDescent(GradientDescent.Method.DICHOTOMY)
    adaptive = GradientDescent(GradientDescent.Method.ADAPTIVE)

    trajectories = []
    labels = []
    for method_name, method in [
        ('Constant learning_rate', constant),
        ('Descending learning_rate', descending),
        ('Optimal learning_rate (gold ratio)', golden_ratio),
        ('Optimal learning_rate (dichotomy)', dichotomy),
        ('Adaptive learning_rate', adaptive)
    ]:
        print(f'{method_name}:')
        point, step, trajectory = method.find_min(f, variables, starting_point)
        labels.append(method_name.replace(' ', '_'))
        trajectories.append(trajectory)
        point_formatted = [{str(var): f'{p:.16f}'} for var, p in zip([x, y, z], point)]
        print(point_formatted, step, "\n")


    xlim = (-10, 10)
    ylim = (-10, 10)

    # plot_3d(sp.lambdify([x, y], f, 'numpy'), trajectories, labels, [x, y], xlim, ylim)

    def quadratic_form(x, Q):
        """Квадратичная форма f(x) = x^T * Q * x."""
        return x.T @ Q @ x


    Q = np.array([
        [1 / 3, 0],
        [0, 1]
    ])

    x0 = np.array([10, 10])  # начальная точка

    result = fmin_bfgs(quadratic_form, x0, args=(Q,), disp=True)
    point_formatted = [{'x': f'{result[0]:.16f}'}, {'y': f'{result[1]:.16f}'}]
    print("Минимум:", point_formatted)


    result = fmin_cg(quadratic_form, x0, args=(Q,), disp=True)
    point_formatted = [{'x': f'{result[0]:.16f}'}, {'y': f'{result[1]:.16f}'}]
    print("Минимум:", point_formatted)


    # result = fmin_ncg(quadratic_form, x0, args=(Q,), disp=True)
    # point_formatted = [{'x': f'{result[0]:.16f}'}, {'y': f'{result[1]:.16f}'}]
    # print("Минимум:", point_formatted)



