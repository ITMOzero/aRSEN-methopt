import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import time
import torch
import torch.nn as nn
import torch.optim as optim
import psutil
from memory_profiler import memory_usage

def sgd(X, y, epochs=100, batch_size=1, lr_schedule=None, reg=None, alpha=0.01, verbose=False):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0.0
    history = []

    for epoch in range(epochs):
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        lr_epoch = lr_schedule(epoch)

        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            y_pred = X_batch @ weights + bias
            error = y_pred - y_batch
            grad_w = X_batch.T @ error / len(y_batch)
            grad_b = np.mean(error)

            match reg:
                case 'l1': grad_w += alpha * np.sign(weights)
                case 'l2': grad_w += alpha * weights
                case 'elastic': grad_w += alpha * (0.5 * weights + 0.5 * np.sign(weights))

            weights -= lr_epoch * grad_w
            bias -= lr_epoch * grad_b

        y_pred_total = X @ weights + bias
        loss = mean_squared_error(y, y_pred_total)
        history.append(loss)

        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}")

    return weights, bias, history


def torch_train(X_train, y_train, X_test, y_test, optimizer_type='SGD', epochs=100, lr=0.01, weight_decay=0.0, momentum=0.1, nesterov=False):
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    model = nn.Linear(X_train.shape[1], 1)
    criterion = nn.MSELoss()

    match optimizer_type:
        case 'SGD': optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)
        case 'Adam': optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        case 'RMSprop': optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
        case 'AdaGrad': optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)

    losses = []
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    duration = time.time() - start_time
    y_pred = model(X_test_tensor).detach().numpy().flatten()

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        'optimizer': optimizer_type,
        'mse': mse,
        'r2': r2,
        'losses': losses,
        'duration': duration
    }


def run_experiment(batch_sizes, regularizations, learning_rate_schedule, epochs=100):
    X, y = make_regression(n_samples=2500, n_features=20, noise=10, random_state=42)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    y = (y - y.mean()) / y.std()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    results = []

    for reg in regularizations:
        for learning_rate in learning_rate_schedule.keys():
            for batch_size in batch_sizes:
                mem_before = psutil.Process().memory_info().rss / (1024 * 1024)
                start_time = time.time()

                weights, bias, history = sgd(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr_schedule=learning_rate_schedule[learning_rate],
                    reg=reg,
                    alpha=0.01
                )

                duration = time.time() - start_time
                mem_after = psutil.Process().memory_info().rss / (1024 * 1024)
                delta_memory = mem_after - mem_before

                y_pred = X_test @ weights + bias
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                results.append({
                    'batch_size': batch_size,
                    'regularization': reg,
                    'mse': mse,
                    'r2': r2,
                    'time': duration,
                    'history': history,
                    'memory_MB': delta_memory
                })

                print(
                    f"[SGD][Batch={batch_size}, Reg={reg}, Lr={learning_rate}] MSE={mse:.4f}, R2={r2:.4f}, Time={duration:.2f}s, RAM Δ={delta_memory:.6f} MB")

    torch_optimizers = [
        {'optimizer_type': 'SGD', 'momentum': 0.0},
        {'optimizer_type': 'SGD', 'momentum': 0.9},
        {'optimizer_type': 'SGD', 'momentum': 0.9, 'nesterov': True},
        {'optimizer_type': 'Adam'},
        {'optimizer_type': 'RMSprop'},
        {'optimizer_type': 'AdaGrad'}
    ]

    for cfg in torch_optimizers:
        def wrapped():
            return torch_train(X_train, y_train, X_test, y_test, **cfg)

        mem_usage = memory_usage((wrapped,), max_iterations=1, interval=0.1)
        result = wrapped()
        result['memory_MB'] = max(mem_usage) - psutil.Process().memory_info().rss / (1024 * 1024)
        results.append(result)

        print(f"[Torch][{result['optimizer']}] MSE={result['mse']:.4f}, R2={result['r2']:.4f}, Time={result['duration']:.2f}s, RAM Δ={result['memory_MB']:.6f} MB")

    return results


def plot_results(results):
    plt.figure(figsize=(12, 6))
    for res in results:
        if 'history' in res:
            label = f"SGD: Batch={res.get('batch_size')}, Reg={res.get('regularization')}"
        else:
            label = f"Torch: {res['optimizer']}"
        plt.plot(res.get('history', res.get('losses', [])), label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Training Loss per Epoch")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    batch_sizes = [1, 32, 128, 1000, 2000]
    regularizations = ['l1', 'l2', 'elastic']
    learning_rate_schedule = {
        'standard': lambda *args: 1e-3,
        'step_decay': lambda x: 0.01 * (0.9 ** (x // 100)),
        'linear_decay': lambda x: max(1e-3, 1e-1 - x * 1e-3)
    }
    results = run_experiment(batch_sizes, regularizations, learning_rate_schedule)
    # plot_results(results)


if __name__ == "__main__":
    main()