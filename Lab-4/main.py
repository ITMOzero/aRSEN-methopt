import math
import random

from functions import *
from graphics import *


def simulated_annealing(f, initial_solution, temperature=1000, cooling_rate=0.95, min_temp=0.01, iterations=1000):
    """
    Выполнение оптимизации методом имитации отжига.

    :param f: Целевая функция для минимизации.
    :param initial_solution: Начальное решение.
    :param temperature: Начальная температура.
    :param cooling_rate: Темп охлаждения.
    :param min_temp: Минимальная температура для завершения.
    :param iterations: Количество итераций.
    :return: Лучшее найденное решение.
    """
    current_solution = initial_solution
    best_solution = initial_solution
    best_value = f(*best_solution)

    trajectory = [best_solution]

    for i in range(iterations):
        if temperature <= min_temp:
            break

        neighbor = current_solution + np.random.uniform(-1, 1, size=current_solution.shape)
        neighbor_value = f(*neighbor)

        current_value = f(*current_solution)

        acceptance_prob = 1 if neighbor_value < current_value else math.exp(
            (current_value - neighbor_value) / temperature)

        if random.random() < acceptance_prob:
            current_solution = neighbor

        if neighbor_value < best_value:
            best_solution = neighbor
            best_value = neighbor_value

        temperature *= cooling_rate
        trajectory.append(current_solution)

    return best_solution, best_value, np.array(trajectory)


def artificial_bee_colony(f, initial_solution, colony_size=3, iterations=100, limit=10):
    """
    Выполнение оптимизации методом искусственной пчелиной колонии (ABC).

    :param f: Целевая функция для минимизации.
    :param initial_solution: Начальное решение.
    :param colony_size: Размер колонии пчел.
    :param iterations: Максимальное количество циклов.
    :param limit: Максимальное количество неудачных попыток поиска.
    :return: Лучшее найденное решение.
    """

    dim = len(initial_solution)
    population = np.random.uniform(low=-10, high=10, size=(colony_size, dim))

    trajectory = [population]

    values = np.array([f(*sol) for sol in population])

    best_solution = population[np.argmin(values)]
    best_value = np.min(values)

    trial = np.zeros(colony_size)

    for cycle in range(iterations):

        for i in range(colony_size):

            new_solution = population[i] + np.random.uniform(-1, 1, size=dim)
            new_value = f(*new_solution)

            if new_value < values[i]:
                population[i] = new_solution
                values[i] = new_value
                trial[i] = 0
            else:
                trial[i] += 1

        current_best_value = np.min(values)
        if current_best_value < best_value:
            best_value = current_best_value
            best_solution = population[np.argmin(values)]

        for i in range(colony_size):
            if trial[i] > limit:
                population[i] = np.random.uniform(low=-10, high=10, size=dim)
                values[i] = f(*population[i])
                trial[i] = 0

        trajectory.append(population.copy())

        print(f"Cycle {cycle + 1}/{iterations}, Best Value: {best_value}")

    return best_solution, best_value, trajectory


if __name__ == "__main__":

    f, variables, initial_solution = select_function('scipy_f2')

    f_lambdified = sp.lambdify(variables, f, 'numpy')

    best_solution, best_value, trajectory = artificial_bee_colony(f_lambdified, np.array(initial_solution))

    if len(initial_solution) == 1:
        path_dict = {'Optimization Path': trajectory}
        animate_bee_colony(f_lambdified, path_dict)
    elif len(initial_solution) == 2:
        animate_bee_3d(
            f_lambdified,
            trajectory,
            variables,
        )
    else:
        print("Can't animate it")

    best_solution, best_value, trajectory = simulated_annealing(f_lambdified, np.array(initial_solution))
    print(f"Best solution: {best_solution}, Value: {best_value}, iterations: {trajectory.size}")

    if len(initial_solution) == 1:
        path_dict = {'Optimization Path': trajectory}
        animate(f_lambdified, path_dict)
    elif len(initial_solution) == 2:
        animate_3d(
            f_lambdified,
            trajectory,
            variables,
            interval=0.05
        )
    else:
        print("Can't animate it")
