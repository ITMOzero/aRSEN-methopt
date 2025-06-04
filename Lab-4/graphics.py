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



def animate_3d(f, trajectory, variables, interval=0.1):
    """
    Анимация для функции f(x, y) в 3D:
    - сначала строим поверхность f на сетке (xlim × ylim),
    - затем анимируем движение по заданной 2D-траектории (trajectory.shape = (N, 2)),
      отображая точку и линию, соединяющую все предыдущие точки траектории.

    :param f: ламбда-функция, принимающая два массива (X, Y) и возвращающая Z = f(X, Y).
    :param trajectory: numpy-массив shape=(N, 2), где каждая строка = [x_i, y_i].
    :param variables: список из двух имен переменных, например ['x', 'y'] (для подписи осей).
    :param xlim: кортеж (xmin, xmax) по оси X.
    :param ylim: кортеж (ymin, ymax) по оси Y.
    :param interval: задержка (в секундах) между кадрами анимации.
    """


    var_names = [str(v) for v in variables]

    xs = trajectory[:, 0]
    ys = trajectory[:, 1]


    dx = (xs.max() - xs.min()) * 0.1
    dy = (ys.max() - ys.min()) * 0.1
    xlim = (xs.min() - dx, xs.max() + dx)
    ylim = (ys.min() - dy, ys.max() + dy)

    x = np.linspace(xlim[0], xlim[1], 100)
    y = np.linspace(ylim[0], ylim[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)


    plt.ion()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')


    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)

    ax.set_xlabel(var_names[0])
    ax.set_ylabel(var_names[1])
    ax.set_zlabel('f({}, {})'.format(var_names[0], var_names[1]))

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


    line_path, = ax.plot([], [], [], 'r-', linewidth=2, label='Траектория')
    point_curr = ax.scatter([], [], [], color='red', s=50, label='Текущая точка')

    ax.legend()


    input("Нажмите Enter, чтобы запустить 3D-анимацию...")


    trajectory = np.array(trajectory)
    N = trajectory.shape[0]


    past_x = []
    past_y = []
    past_z = []

    for i in range(N):
        x_i, y_i = trajectory[i]
        z_i = f(np.array([[x_i]]), np.array([[y_i]]))[0, 0] if hasattr(f, "__call__") else f(x_i, y_i)

        past_x.append(x_i)
        past_y.append(y_i)
        past_z.append(z_i)

        line_path.set_data(past_x, past_y)
        line_path.set_3d_properties(past_z)

        point_curr._offsets3d = ([x_i], [y_i], [z_i])

        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(interval)

    print("Анимация завершена.")
    plt.ioff()
    plt.show()




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



def animate_bee_3d(f, trajectory, variables, interval=0.1):
    """
    3D‐анимация для метода искусственной пчелиной колонии (ABC) на функции f(x,y).
    trajectory: список длины T, где каждый элемент — массив shape=(colony_size, 2)
                с координатами [x, y] всех пчёл на данном шаге.
    variables: список из двух символов, например ['x', 'y'], для подписей осей.
    interval: задержка между кадрами анимации в секундах.
    """


    trajectory = np.array(trajectory)
    T, colony_size, dim = trajectory.shape
    if dim != 2:
        print("Ошибка: функция animate_bee_3d рассчитана только на двумерное пространство (dim=2).")
        return


    var_names = [str(v) for v in variables]

    all_x = trajectory[:, :, 0].flatten()
    all_y = trajectory[:, :, 1].flatten()


    dx = (all_x.max() - all_x.min()) * 0.1
    dy = (all_y.max() - all_y.min()) * 0.1
    xlim = (all_x.min() - dx, all_x.max() + dx)
    ylim = (all_y.min() - dy, all_y.max() + dy)


    x = np.linspace(xlim[0], xlim[1], 100)
    y = np.linspace(ylim[0], ylim[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)


    plt.ion()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')


    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)


    ax.set_xlabel(var_names[0])
    ax.set_ylabel(var_names[1])
    ax.set_zlabel(f"f({var_names[0]}, {var_names[1]})")

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


    history_lines = []
    current_points = []
    colors = plt.cm.tab10(np.linspace(0, 1, colony_size))  # разные цвета для каждой пчелы

    for j in range(colony_size):

        line_j, = ax.plot([], [], [], color=colors[j], linewidth=2, label=f'Пчела {j+1} (траектория)')
        history_lines.append(line_j)

        point_j = ax.scatter([], [], [], color=colors[j], s=50, marker='o',
                             label=f'Пчела {j+1} (текущая)')
        current_points.append(point_j)

    ax.legend()

    input("Нажмите Enter, чтобы запустить 3D-анимацию для пчелиной колонии...")


    past_coords = [[], [], [], [], [], [], [], [], [], []]

    for t in range(T):
        population_t = trajectory[t]

        for j in range(colony_size):
            x_j, y_j = population_t[j]

            z_j = f(np.array([[x_j]]), np.array([[y_j]]))[0, 0] \
                  if hasattr(f, "__call__") else f(x_j, y_j)

            past_coords[j].append((x_j, y_j, z_j))


            xs_line = [pt[0] for pt in past_coords[j]]
            ys_line = [pt[1] for pt in past_coords[j]]
            zs_line = [pt[2] for pt in past_coords[j]]


            history_lines[j].set_data(xs_line, ys_line)
            history_lines[j].set_3d_properties(zs_line)

            current_points[j]._offsets3d = ([x_j], [y_j], [z_j])

        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(interval)

    print("done")
    plt.ioff()
    plt.show()