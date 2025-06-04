import time
import matplotlib.pyplot as plt
import numpy as np
def animate(f, path_dict, bounds=(-10, 10), num_points=400, interval=0.1):
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


def animate_bee_colony(f, path_dict, bounds=(-10, 10), num_points=400, interval=0.1):
    """
    Анимация для метода искусственной пчелиной колонии,
    где trajectory — список массивов shape=(colony_size, ).
    На каждом шаге рисуются сразу все пчёлы (точки) и вся их история.
    """
    # Построим график функции
    x_vals = np.linspace(bounds[0], bounds[1], num_points)
    y_vals = [f(x) for x in x_vals]

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_vals, y_vals, label='f(x)', color='black', zorder=1)
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title('Минимизация функции f(x) — движение колонии пчёл')
    ax.grid(True)

    scatters = {}        # для текущих позиций пчёл
    past_scatters = {}   # для исторических позиций

    # Инициализация пустых scatter-объектов
    for label, path in path_dict.items():
        # path — список массивов shape=(colony_size,)
        scatters[label] = ax.scatter([], [], s=80, zorder=4, label=label + ' (текущие)')
        past_scatters[label] = ax.scatter([], [], s=20, alpha=0.5, zorder=3, label=label + ' (прошлые)')

    ax.legend()

    input("Нажмите Enter, чтобы начать анимацию...")

    max_length = max(len(path) for path in path_dict.values())
    for i in range(max_length):
        for label, path in path_dict.items():
            if i < len(path):
                # Текущая популяция пчёл на шаге i
                current_population = path[i]
                # Если это массив shape=(colony_size, 1), сплющим до (colony_size,)
                current_population = current_population.flatten()

                # Вычисляем координаты y для всех пчёл в текущей популяции
                x_current = current_population
                y_current = np.array([f(x) for x in x_current])

                # Готовим все прошлые позиции (до шага i включительно)
                all_past_x = []
                all_past_y = []
                for k in range(i + 1):
                    pop_k = path[k].flatten()
                    all_past_x.append(pop_k)
                    all_past_y.append([f(x) for x in pop_k])
                all_past_x = np.concatenate(all_past_x)
                all_past_y = np.concatenate(all_past_y)

                # Обновляем scatter'ы
                past_scatters[label].set_offsets(np.c_[all_past_x, all_past_y])
                scatters[label].set_offsets(np.c_[x_current, y_current])

        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(interval)

    print('Анимация завершена.')
    plt.ioff()
    plt.show()