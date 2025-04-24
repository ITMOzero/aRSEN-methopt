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