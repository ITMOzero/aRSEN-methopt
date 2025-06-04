import time
import matplotlib.pyplot as plt
import numpy as np
def animate(f, path_dict, num_points=400, interval=0.1):

    trajectory = list(path_dict.values())[0]
    finite_vals = [x for x in trajectory if np.isfinite(x)]
    if finite_vals:
        b = max(abs(max(finite_vals)), abs(min(finite_vals)))
        bounds = (-b, b)
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

        for label, path in path_dict.items():
            scatters[label] = ax.scatter([], [], s=80, zorder=4, label=label)
            past_scatters[label] = ax.scatter([], [], s=20, alpha=0.5, zorder=3)

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

            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(interval)

        print('done')
        plt.ioff()
        plt.show()
    else:
        print("incorrect values")




def animate_bee_colony(f, path_dict, num_points=400, interval=0.1):
    """
    Анимация для метода искусственной пчелиной колонии,
    где trajectory — список массивов shape=(colony_size, ).
    На каждом шаге рисуются сразу все пчёлы (точки) и вся их история.
    """
    trajectory = list(path_dict.values())[0]
    finite_vals = [point for population in trajectory for point in population if np.isfinite(f(*point))]

    if finite_vals:
        b = max(abs(max(finite_vals)), abs(min(finite_vals)))
        bounds = (-b, b)
        x_vals = np.linspace(bounds[0], bounds[1], num_points)
        y_vals = [f(x) for x in x_vals]

        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x_vals, y_vals, label='f(x)', color='black', zorder=1)
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title('Минимизация функции f(x) — движение колонии пчёл')
        ax.grid(True)

        scatters = {}
        past_scatters = {}

        for label, path in path_dict.items():
            scatters[label] = ax.scatter([], [], s=80, zorder=4, label=label + ' (текущие)')
            past_scatters[label] = ax.scatter([], [], s=20, alpha=0.5, zorder=3, label=label + ' (прошлые)')

        ax.legend()

        input("Нажмите Enter, чтобы начать анимацию...")

        max_length = max(len(path) for path in path_dict.values())
        for i in range(max_length):
            for label, path in path_dict.items():
                if i < len(path):

                    current_population = path[i]

                    current_population = current_population.flatten()

                    x_current = current_population
                    y_current = np.array([f(x) for x in x_current])

                    all_past_x = []
                    all_past_y = []
                    for k in range(i + 1):
                        pop_k = path[k].flatten()
                        all_past_x.append(pop_k)
                        all_past_y.append([f(x) for x in pop_k])
                    all_past_x = np.concatenate(all_past_x)
                    all_past_y = np.concatenate(all_past_y)

                    past_scatters[label].set_offsets(np.c_[all_past_x, all_past_y])
                    scatters[label].set_offsets(np.c_[x_current, y_current])

            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(interval)

        print('done')
        plt.ioff()
        plt.show()
    else:
        print("incorrect values")