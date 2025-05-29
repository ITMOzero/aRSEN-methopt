import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import pandas as pd

# Генерация данных
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Нормализация
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Инициализация весов
def initialize_weights(n_features):
    return np.zeros(n_features)

# Функция потерь (MSE)
# Mean Squared Error
def compute_loss(X, y, w):
    y_pred = X.dot(w)
    return np.mean((y_pred - y) ** 2)

# Градиент MSE
def compute_gradient(X, y, w):
    y_pred = X.dot(w)
    error = y_pred - y
    return 2 * X.T.dot(error) / len(y)

# Реализация Ranger (Lookahead + Adam)
class Ranger:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, k=5, alpha=0.5):
        self.lr = lr  # learning rate
        # Коэффициент для моментума (экспоненциальное скользящее среднее градиентов).
        # Чем ближе к 1, тем больше учитываются прошлые градиенты (сглаживание).
        self.beta1 = beta1  # для момента
        # Коэффициент для адаптивного шага (экспоненциальное скользящее среднее квадратов градиентов).
        # Чем ближе к 1, тем плавнее меняется learning rate.
        self.beta2 = beta2  # для адаптивного шага
        # Чем реже, тем стабильнее, но медленнее сходимость.
        self.k = k  # частота обновления Lookahead
        # Вес Lookahead (доля, на которую "медленные" веса обновляются в сторону "быстрых")
        # Чем больше, тем агрессивнее корректировка весов.
        self.alpha = alpha  # вес Lookahead
        self.m = None  # момент
        self.v = None  # адаптивный момент
        self.t = 0  # счётчик шагов
        self.slow_weights = None  # медленные веса для Lookahead
        self.base_optimizer = "adam" if beta1 > 0 or beta2 > 0 else "sgd"

    def update(self, w, grad):
        if self.m is None:
            self.m = np.zeros_like(w)
            self.v = np.zeros_like(w)
            self.slow_weights = np.copy(w)

        self.t += 1

        if self.base_optimizer == "adam":
            self.m = self.beta1 * self.m + (1 - self.beta1) * grad
            self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
            m_hat = self.m / (1 - self.beta1 ** self.t)
            v_hat = self.v / (1 - self.beta2 ** self.t)
            w -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        else:  # SGD
            w -= self.lr * grad

        # Adam
        # self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        # self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        # m_hat = self.m / (1 - self.beta1 ** self.t)
        # v_hat = self.v / (1 - self.beta2 ** self.t)
        # w -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)

        # Lookahead (каждые k шагов)
        if self.alpha > 0 and self.t % self.k == 0:
            self.slow_weights += self.alpha * (w - self.slow_weights)
            w = np.copy(self.slow_weights)

        return w

# Обучение модели
def train(X, y, optimizer, epochs=100, batch_size=32):
    n_samples, n_features = X.shape
    w = initialize_weights(n_features)
    losses = []

    for epoch in range(epochs):
        # Мини-батчи
        indices = np.random.permutation(n_samples)
        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i + batch_size]
            X_batch, y_batch = X[batch_idx], y[batch_idx]

            grad = compute_gradient(X_batch, y_batch, w)
            w = optimizer.update(w, grad)

        loss = compute_loss(X, y, w)
        losses.append(loss)
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

    return w, losses

# Сравнение оптимизаторов
optimizers = {
    "SGD": Ranger(lr=0.0017, beta1=0, beta2=0, k=1, alpha=0), # Вырожденный случай SGD
    "Adam": Ranger(lr=0.1, beta1=0.9, beta2=0.9, k=1, alpha=0), # Только Adam
    "Ranger": Ranger(lr=0.15, beta1=0.4, beta2=0.6, k=5, alpha=0.5)  # Полный Ranger
}

# optimizers = { # correct values in result, wrong grafics
#     "SGD": Ranger(lr=0.003, beta1=0, beta2=0, k=1, alpha=0), # Вырожденный случай SGD
#     "Adam": Ranger(lr=0.1, beta1=0.9, beta2=0.9, k=1, alpha=0), # Только Adam
#     "Ranger": Ranger(lr=0.15, beta1=0.4, beta2=0.6, k=5, alpha=0.5)  # Полный Ranger
# }

results = {}
for name, opt in optimizers.items():
    print(f"\nTraining with {name}...")
    w, losses = train(X_train, y_train, opt, epochs=50)
    results[name] = losses

# График потерь
plt.figure(figsize=(10, 6))
for name, losses in results.items():
    plt.plot(losses, label=name)
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.title("Сравнение оптимизаторов")
plt.savefig("optimizers_comparison.png")
plt.close()

# -----------------------------------------------------------


# Словарь для хранения статистики
stats = {
    "Optimizer": [],
    "Final Loss": [],
    "Min Loss": [],
    "Epochs to Min Loss": [],
    "Time per Epoch (ms)": []
}

# Запуск обучения и сбор статистики
results = {}
for name, opt in optimizers.items():
    print(f"\nTraining with {name}...")
    start_time = time.time()
    w, losses = train(X_train, y_train, opt, epochs=50)
    training_time = (time.time() - start_time) * 1000 / 50  # мс на эпоху

    results[name] = losses
    stats["Optimizer"].append(name)
    stats["Final Loss"].append(losses[-1])
    stats["Min Loss"].append(min(losses))
    stats["Epochs to Min Loss"].append(np.argmin(losses) + 1)
    stats["Time per Epoch (ms)"].append(training_time)

# Создаем DataFrame
df_results = pd.DataFrame(stats)


# Функция для подсветки минимальных значений
def highlight_min(s):
    is_min = s == s.min()
    return ['background-color: lightgreen' if v else '' for v in is_min]


# Применяем стилизацию
styled_df = df_results.style.apply(highlight_min,
                                   subset=["Final Loss", "Min Loss", "Time per Epoch (ms)"])

print("\n" + "=" * 50)
print("ИТОГОВАЯ СВОДКА ПО ОПТИМИЗАТОРАМ")
print("=" * 50)
print(df_results.to_string(index=False))

print("=" * 50)

# График сходимости
plt.figure(figsize=(10, 6))
for name, losses in results.items():
    plt.plot(losses, label=name)
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.title("Сравнение оптимизаторов")
plt.grid(True)
plt.savefig("loss_comparison.png")
plt.close()



# w_values = np.linspace(-5, 5, 100)
# losses = [compute_loss(X, y, np.array([w])) for w in w_values]
#
# plt.plot(w_values, losses)
# plt.xlabel("Вес w")
# plt.ylabel("Loss L(w)")
# plt.title("Градиентный спуск: движение к минимуму")
# plt.savefig("1D_model.gif")
# plt.close()