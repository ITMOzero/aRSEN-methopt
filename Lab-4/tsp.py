import random
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance_matrix


def generate_cities(n_cities: int, seed: int = 42) -> np.ndarray:
    np.random.seed(seed)
    return np.random.rand(n_cities, 2) * 100


class GeneticAlgorithmTSPVisual:
    def __init__(self, cities: np.ndarray, population_size: int = 100,
                 mutation_rate: float = 0.01, elite_size: int = 20,
                 generations: int = 500, plot_interval: int = 10):
        self.cities = cities
        self.n_cities = len(cities)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.generations = generations
        self.plot_interval = plot_interval
        self.dist_matrix = distance_matrix(cities, cities)
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        plt.show(block=False)

    def create_individual(self) -> List[int]:
        return random.sample(range(self.n_cities), self.n_cities)

    def initial_population(self) -> List[List[int]]:
        return [self.create_individual() for _ in range(self.population_size)]

    def fitness(self, individual: List[int]) -> float:
        total_distance = 0
        for i in range(self.n_cities - 1):
            total_distance += self.dist_matrix[individual[i], individual[i + 1]]
        total_distance += self.dist_matrix[individual[-1], individual[0]]
        return 1 / total_distance

    def rank_population(self, population: List[List[int]]) -> List[Tuple[float, List[int]]]:
        return sorted([(self.fitness(ind), ind) for ind in population], key=lambda x: x[0], reverse=True)

    def selection(self, ranked_population: List[Tuple[float, List[int]]]) -> List[List[int]]:
        fitness_values = [f for f, _ in ranked_population]
        total_fitness = sum(fitness_values)
        probabilities = [f / total_fitness for f in fitness_values]
        elite = [ind for _, ind in ranked_population[:self.elite_size]]
        selected = elite + [ranked_population[np.random.choice(
            len(ranked_population), p=probabilities)][1]
                            for _ in range(self.population_size - self.elite_size)]
        return selected

    def crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        size = len(parent1)
        child = [-1] * size
        start, end = sorted(random.sample(range(size), 2))
        child[start:end] = parent1[start:end]
        remaining = [city for city in parent2 if city not in child]
        child[:start] = remaining[:start]
        child[end:] = remaining[start:]
        return child

    def mutate(self, individual: List[int]) -> List[int]:
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(self.n_cities), 2)
            individual[i], individual[j] = individual[j], individual[i]
        return individual

    def plot_current_route(self, individual: List[int], generation: int, distance: float):
        self.ax.clear()
        self.ax.scatter(self.cities[:, 0], self.cities[:, 1], c='red', s=100)
        route = individual + [individual[0]]
        self.ax.plot(self.cities[route, 0], self.cities[route, 1], 'b-')
        for i, (x, y) in enumerate(self.cities):
            self.ax.text(x, y, str(i), fontsize=12)
        self.ax.set_title(f"Поколение {generation}. Длина маршрута: {distance:.2f}")
        self.ax.set_xlabel("X координата")
        self.ax.set_ylabel("Y координата")
        self.ax.grid(True)
        plt.draw()
        plt.pause(0.1)

    def run(self, verbose: bool = True) -> Tuple[List[int], float]:
        input("Нажмите Enter чтобы запустить анимацию...")
        population = self.initial_population()
        best_fitness = 0
        best_individual = None
        fitness_history = []

        for generation in range(self.generations):
            population = self.evolve(population)
            current_best_fitness, current_best = max(self.rank_population(population))

            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = current_best.copy()

            distance = 1 / best_fitness if best_fitness > 0 else float('inf')
            fitness_history.append(distance)

            if generation % self.plot_interval == 0:
                self.plot_current_route(best_individual, generation, distance)

            if verbose and generation % 50 == 0:
                print(f"Поколение {generation}: Лучшая длина маршрута = {distance:.2f}")

        return best_individual, 1 / best_fitness, fitness_history

    def evolve(self, population: List[List[int]]) -> List[List[int]]:
        ranked_population = self.rank_population(population)
        selected = self.selection(ranked_population)
        children = []
        for i in range(len(selected) - self.elite_size):
            parent1, parent2 = random.sample(selected, 2)
            child = self.crossover(parent1, parent2)
            children.append(child)
        elite = [ind for _, ind in ranked_population[:self.elite_size]]
        next_generation = elite + children
        next_generation = [self.mutate(ind) for ind in next_generation]
        return next_generation


def main():
    n_cities = 15
    plot_interval = 5

    cities = generate_cities(n_cities)

    print("Города сгенерированы. Окно визуализации появится сейчас.")
    print("Когда будете готовы, вернитесь в консоль и...")

    ga = GeneticAlgorithmTSPVisual(
        cities,
        population_size=100,
        mutation_rate=0.02,
        elite_size=15,
        generations=200,
        plot_interval=plot_interval
    )

    best_route, best_distance, fitness_history = ga.run()

    print(f"\nЛучший найденный маршрут имеет длину: {best_distance:.2f}")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(cities[:, 0], cities[:, 1], c='red', s=100)
    route = best_route + [best_route[0]]
    plt.plot(cities[route, 0], cities[route, 1], 'b-')
    for i, (x, y) in enumerate(cities):
        plt.text(x, y, str(i), fontsize=12)
    plt.title(f"Финальный маршрут (длина = {best_distance:.2f})")
    plt.xlabel("X координата")
    plt.ylabel("Y координата")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(fitness_history)
    plt.title("Сходимость генетического алгоритма")
    plt.xlabel("Поколение")
    plt.ylabel("Длина лучшего маршрута")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
