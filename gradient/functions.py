import sympy as sp


def select_function(f_type):
    """
    Выбор функции для минимизации.
    В зависимости от типа функции возвращает соответствующее выражение для минимизации и начальные точки.

    :param f_type: тип функции для минимизации
    :return: кортеж из:
        - выражение для минимизации (SymPy выражение),
        - список переменных, для которых проводится минимизация,
        - список начальных точек для минимизации
    """
    x, y, z = sp.symbols('x y z')

    if f_type == 'f1':

        # Функция f1: f(x) = x^2 + x/3 - 5
        return x ** 2 + x / 3 - 5, [x], [10]

    elif f_type == 'f2':

        # Функция f2: f(x) = 3 * x^4 - 3 * x^3 - x^2 - x
        return 3 * x ** 4 - 3 * x ** 3 - x ** 2 - x, [x], [10]

    elif f_type == 'f_himmelblau':

        # Функция Химмельблау: f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
        return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2, [x, y], [10, 10]

    elif f_type == 'scipy_f1':

        # Функция для анализа scipy: f(x, y) = x^3 + 2 * x^2 + 3 * x - 10
        return x ** 3 + 2 * x ** 2 + 3 * x - 10, [x], [10]

    elif f_type == 'scipy_f2':

        # Функция для анализа scipy: f(x, y) = f = y ^ 2 + x ^ 2 / 3
        return y ** 2 + x ** 2 / 3 + 5, [x, y], [10, 10]

    else:
        raise ValueError(f"Unknown function type: {f_type}")

