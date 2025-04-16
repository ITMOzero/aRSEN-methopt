import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from optimization_methods import OptimizationMethods
from scipy.optimize import rosen, rosen_der, rosen_hess
import os

# Демонстрация работы методов квазиНьютоновских методов на фото

def ensure_dir(directory):
    """Создает директорию, если она не существует"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def plot_optimization_process(histories, labels, save_path=None):
    """Визуализация процесса оптимизации"""
    plt.figure(figsize=(12, 8))

    # Создание сетки для графика функции
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.array([rosen([xi, yi]) for xi, yi in zip(X.ravel(), Y.ravel())]).reshape(X.shape)

    # Контурный график функции
    levels = np.logspace(-1, 3, 20)
    plt.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.5)
    plt.plot(1, 1, 'r*', markersize=10, label='Global minimum')

    # Траектории оптимизации
    colors = ['r', 'g', 'b', 'm', 'c']
    for i, (history, label) in enumerate(zip(histories, labels)):
        plt.plot(history[:, 0], history[:, 1], 'o-', color=colors[i],
                 markersize=4, linewidth=1.5, label=label)
        plt.plot(history[0, 0], history[0, 1], 'ko', markersize=6, label='Start' if i == 0 else "")

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Optimization trajectories on Rosenbrock function')
    plt.legend()
    plt.grid(True)
    plt.colorbar(label='Function value')

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_convergence(histories, labels, save_path=None):
    """График сходимости методов"""
    plt.figure(figsize=(12, 6))

    for i, (history, label) in enumerate(zip(histories, labels)):
        f_values = [rosen(x) for x in history]
        plt.semilogy(f_values, 'o-', label=label, markersize=4)

    plt.xlabel('Iteration')
    plt.ylabel('Function value (log scale)')
    plt.title('Convergence of optimization methods')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_3d_surface(save_path=None):
    """3D визуализация функции Розенброка"""
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')

    x = np.linspace(-2, 2, 100)
    y = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.array([rosen([xi, yi]) for xi, yi in zip(X.ravel(), Y.ravel())]).reshape(X.shape)

    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    ax.set_title('Rosenbrock function 3D visualization')
    fig.colorbar(surf, shrink=0.5, aspect=5)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def compare_methods():
    """Сравнение методов оптимизации"""
    # Создаем директорию для сохранения графиков
    output_dir = "optimization_plots"
    ensure_dir(output_dir)

    # Начальная точка
    x0 = np.array([-1.2, 1.0])

    # Оптимизация разными методами
    methods = [
        ("Newton-CG", OptimizationMethods.newton_cg, rosen, rosen_der, rosen_hess),
        ("BFGS", OptimizationMethods.bfgs, rosen, rosen_der, None),
        ("Modified BFGS", OptimizationMethods.modified_bfgs, rosen, rosen_der, None)
    ]

    results = []
    histories = []
    labels = []

    for name, method, f, grad, hess in methods:
        if hess:
            x_opt, history = method(f, grad, hess, x0)
        else:
            x_opt, history = method(f, grad, x0)

        n_iter = len(history) - 1
        final_value = f(x_opt)
        results.append((name, n_iter, final_value))
        histories.append(history)
        labels.append(name)

        print(f"{name}:")
        print(f"  Iterations: {n_iter}")
        print(f"  Final point: {x_opt}")
        print(f"  Final value: {final_value:.6f}")
        print(f"  Convergence: {np.linalg.norm(x_opt - np.array([1.0, 1.0])):.6f}")
        print()

    # Визуализация и сохранение графиков
    plot_3d_surface(os.path.join(output_dir, "rosenbrock_3d.png"))
    plot_optimization_process(histories, labels, os.path.join(output_dir, "optimization_paths.png"))
    plot_convergence(histories, labels, os.path.join(output_dir, "convergence.png"))

    # Таблица результатов
    print("\nComparison table:")
    print("{:<15} {:<15} {:<15}".format("Method", "Iterations", "Final value"))
    print("-" * 45)
    for name, n_iter, final_value in results:
        print("{:<15} {:<15} {:<15.6f}".format(name, n_iter, final_value))

    print(f"\nGraphs saved to directory: {output_dir}")


if __name__ == "__main__":
    compare_methods()






