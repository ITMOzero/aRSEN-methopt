import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from optimization_methods import OptimizationMethods
from scipy.optimize import rosen, rosen_der, rosen_hess
import os

# Демонстрация работы методов квазиНьютоновских методов на gif файлах

def create_optimization_animation(method_name, method_func, f, grad, hess=None, x0=np.array([-1.2, 1.0])):
    """Создает анимацию процесса оптимизации для заданного метода"""
    # Выполняем оптимизацию и получаем историю
    if hess:
        x_opt, history = method_func(f, grad, hess, x0)
    else:
        x_opt, history = method_func(f, grad, x0)

    # Проверяем, что история содержит достаточно точек
    if len(history) < 2:
        print(f"Not enough points in history for {method_name}. Skipping animation.")
        return

    # Преобразуем историю в numpy array для надежности
    history = np.array(history)

    # Подготовка данных для графика
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.array([rosen([xi, yi]) for xi, yi in zip(X.ravel(), Y.ravel())]).reshape(X.shape)

    # Создаем фигуру
    fig, ax = plt.subplots(figsize=(10, 8))

    # Контурный график функции
    levels = np.logspace(-1, 3, 20)
    contour = ax.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.5)
    plt.colorbar(contour, ax=ax, label='Function value')
    ax.plot(1, 1, 'r*', markersize=15, label='Global minimum')

    # Начальная точка
    point, = ax.plot([], [], 'ko', markersize=8, label='Current point')
    line, = ax.plot([], [], 'r-', linewidth=1.5, alpha=0.6, label='Path')

    # Текст с номером итерации
    iter_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    value_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)

    # Настройки графика
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1, 3)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'{method_name} optimization process')
    ax.legend()
    ax.grid(True)

    # Функция инициализации
    def init():
        point.set_data([], [])
        line.set_data([], [])
        iter_text.set_text('')
        value_text.set_text('')
        return point, line, iter_text, value_text

    # Функция анимации
    def update(frame):
        # Обновляем текущую точку
        point.set_data([history[frame, 0]], [history[frame, 1]])

        # Обновляем путь
        line.set_data(history[:frame + 1, 0], history[:frame + 1, 1])

        # Обновляем текст
        iter_text.set_text(f'Iteration: {frame}')
        value_text.set_text(f'Value: {f(history[frame]):.4f}')

        return point, line, iter_text, value_text

    # Создаем анимацию
    frames = len(history)
    anim = FuncAnimation(
        fig, update, frames=frames,
        init_func=init, blit=True, interval=300, repeat_delay=2000
    )

    # Сохраняем анимацию
    output_dir = "optimization_animations"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{method_name.lower().replace(' ', '_')}_animation.gif")

    try:
        # Используем PillowWriter для сохранения
        writer = PillowWriter(fps=5)
        anim.save(filename, writer=writer)
        print(f"Animation successfully saved to {filename}")
    except Exception as e:
        print(f"Failed to save animation: {str(e)}")
    finally:
        plt.close(fig)


def create_all_animations():
    """Создает анимации для всех методов"""
    methods = [
        ("Newton-CG", OptimizationMethods.newton_cg, rosen, rosen_der, rosen_hess),
        ("BFGS", OptimizationMethods.bfgs, rosen, rosen_der, None),
        ("Modified BFGS", OptimizationMethods.modified_bfgs, rosen, rosen_der, None)
    ]

    for name, method, f, grad, hess in methods:
        print(f"\nCreating animation for {name}...")
        create_optimization_animation(name, method, f, grad, hess)


if __name__ == "__main__":
    create_all_animations()