from matplotlib import pyplot as plt


def plot_3d(f, trajectory_constant, trajectory_descending, trajectory_optimal, trajectory_dichotomy, trajectory_adaptive, variables, xlim, ylim):
    plt.figure(figsize=(10, 8))


    ax = plt.axes(projection='3d')



    ax.plot(trajectory_constant[:, 0], trajectory_constant[:, 1], f(trajectory_constant[:, 0], trajectory_constant[:, 1]), 'r-', label='Constant learning_rate')
    ax.plot(trajectory_descending[:, 0], trajectory_descending[:, 1], f(trajectory_descending[:, 0], trajectory_descending[:, 1]), 'g-', label='Descending learning_rate')
    ax.plot(trajectory_optimal[:, 0], trajectory_optimal[:, 1], f(trajectory_optimal[:, 0], trajectory_optimal[:, 1]), 'b-', label='Optimal learning_rate (gold ratio)')
    ax.plot(trajectory_dichotomy[:, 0], trajectory_dichotomy[:, 1], f(trajectory_dichotomy[:, 0], trajectory_dichotomy[:, 1]), 'y-', label='Optimal learning_rate (dichotomy)')
    # ax.plot(trajectory_adaptive[:, 0], trajectory_adaptive[:, 1],f(trajectory_adaptive[:, 0], trajectory_adaptive[:, 1]), 'o-', label='Optimal learning_rate (adaptive)')

    ax.set_xlabel(variables[0])
    ax.set_ylabel(variables[1])
    ax.set_zlabel('f(x, y)')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.legend()

    plt.savefig('gradient_descent_3d_plot.png')