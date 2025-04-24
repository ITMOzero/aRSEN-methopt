import time

import numpy as np
import matplotlib.pyplot as plt


def plot_3d(f, trajectories, labels, variables, xlim, ylim):
    x = np.linspace(xlim[0], xlim[1], 50)
    y = np.linspace(ylim[0], ylim[1], 50)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    for i, (trajectory, label) in enumerate(zip(trajectories, labels)):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5)
        ax.plot(trajectory[:, 0], trajectory[:, 1], f(trajectory[:, 0], trajectory[:, 1]), 'b-', label=label)
        ax.scatter(trajectory[:, 0], trajectory[:, 1], f(trajectory[:, 0], trajectory[:, 1]),
                   facecolors='none', edgecolors='black', s=50, label='Kinks')

        ax.set_xlabel(variables[0])
        ax.set_ylabel(variables[1])
        ax.set_zlabel('f(x, y)')

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.legend()

        plt.savefig(f'gradient_descent_3d_plot_{label}.png')
        plt.show()


def plot_2d(f, trajectories, labels, variables, xlim):
    x = np.linspace(xlim[0], xlim[1], 50)
    y = f(x)
    for i, (trajectory, label) in enumerate(zip(trajectories, labels)):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)

        ax.plot_surface(x, y, cmap='viridis', alpha=0.5)
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', label=label)
        ax.scatter(trajectory[:, 0], trajectory[:, 1], f(trajectory[:, 0], trajectory[:, 1]),
                   facecolors='none', edgecolors='black', s=50, label='Kinks')

        ax.set_xlabel(variables[0])
        ax.set_ylabel('f(x)')

        ax.set_xlim(xlim)
        ax.legend()

        plt.savefig(f'gradient_descent_2d_plot_{label}.png')
        plt.show()


def animate_2d(f, trajectory, xlim):
    x_plt = np.linspace(xlim[0], xlim[1], 50)
    y_plt = f(x_plt)

    plt.ion()
    fig, ax = plt.subplots()
    ax.grid(True)
    ax.plot(x_plt, y_plt)

    point = ax.scatter(trajectory[0][0], f(trajectory[0][0]))

    input('press to start animation')
    for x in trajectory:
        point.set_offsets([x[0], f(x[0])])
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.02)

    plt.ioff()

    ax.scatter(x, f(x), c='blue')
    plt.show()
