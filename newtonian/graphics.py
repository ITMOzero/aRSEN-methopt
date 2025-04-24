import time

import matplotlib.pyplot as plt
import numpy as np


def plot_function(f, path_dict: dict, bounds=(-10, 10), num_points=400):
    x_vals = np.linspace(bounds[0], bounds[1], num_points)
    y_vals = [f(x) for x in x_vals]

    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, y_vals, label='f(x)', color='black')
    for label, path in path_dict.items():
        y_path = [f(x) for x in path]
        plt.plot(path, y_path, '-o', label=label)

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Минимизация функции f(x)')
    plt.legend()
    plt.grid(True)

    plt.savefig(f'plots/graphics_of_{"_".join(str(v) for v in path_dict.keys())}.png', dpi=300)
    plt.show()


def animate(f, path_dict, bounds=(-10, 10), num_points=400, interval=0.1, tangent_length=1.0):
    x_vals = np.linspace(bounds[0], bounds[1], num_points)
    y_vals = [f(x) for x in x_vals]

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_vals, y_vals, label='f(x)', color='black', zorder=1)
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title('Минимизация функции f(x) с касательными')
    ax.grid(True)

    scatters = {}
    past_scatters = {}
    tangent_lines = {}

    for label, path in path_dict.items():
        past_scatters[label] = ax.scatter([], [], s=20, alpha=0.5, zorder=3)
        tangent_lines[label] = ax.plot([], [], '--', alpha=0.7, linewidth=1, zorder=2)[0]

    ax.legend()

    input("Нажмите Enter, чтобы начать анимацию...")

    max_length = max(len(path) for path in path_dict.values())
    for i in range(max_length):
        for label, path in path_dict.items():
            if i < len(path):
                x_point = path[i]
                y_point = f(x_point)

                x_past = path[:i + 1]
                y_past = [f(x) for x in x_past]

                past_scatters[label].set_offsets(np.c_[x_past, y_past])

                scatters[label].set_offsets(np.c_[x_point, y_point])

                h = 1e-5
                derivative = (f(x_point + h) - f(x_point - h)) / (2 * h)
                x_tangent = np.linspace(x_point - tangent_length, x_point + tangent_length, 2)
                y_tangent = y_point + derivative * (x_tangent - x_point)
                tangent_lines[label].set_data(x_tangent, y_tangent)

        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(interval)

    print('done')
    plt.ioff()
    plt.show()