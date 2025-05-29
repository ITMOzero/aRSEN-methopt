#  ================  TRASH  ================


from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ranger import Ranger
from ranger import train
import numpy as np
import pandas as pd
import time

# Загрузка данных
data = fetch_california_housing()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Нормализация
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Обучение с Ranger
opt_ranger = Ranger(lr=0.001, beta1=0.9, beta2=0.999, k=5, alpha=0.5)
w_ranger, losses_ranger = train(X_train, y_train, opt_ranger, epochs=100)

# Оценка на тесте
y_pred = X_test.dot(w_ranger)
mse = np.mean((y_pred - y_test) ** 2)
print(f"\nTest MSE with Ranger: {mse:.4f}")



# Словарь для хранения статистики
stats = {
    "Optimizer": [],
    "Final Loss": [],
    "Min Loss": [],
    "Epochs to Min Loss": [],
    "Time per Epoch (ms)": []
}


for name, opt in optimizers.items():
    print(f"\nTraining with {name}...")
    start_time = time.time()
    w, losses = train(X_train, y_train, opt, epochs=50)
    training_time = (time.time() - start_time) * 1000 / 50  # Среднее время на эпоху в мс

    # Сбор статистики
    stats["Optimizer"].append(name)
    stats["Final Loss"].append(losses[-1])
    stats["Min Loss"].append(min(losses))
    stats["Epochs to Min Loss"].append(np.argmin(losses) + 1)
    stats["Time per Epoch (ms)"].append(training_time)

# Создаем DataFrame и выводим таблицу
df_results = pd.DataFrame(stats)
print("\n" + "=" * 50)
print("ИТОГОВАЯ СВОДКА ПО ОПТИМИЗАТОРАМ")
print("=" * 50)
print(df_results.to_string(index=False))
print("=" * 50)