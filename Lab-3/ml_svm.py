from sklearn import datasets, svm
import matplotlib.pyplot as plt
import numpy as np

# Загрузка данных
iris = datasets.load_iris()
X = iris.data[:, :2]  # Берём только 2 признака для визуализации
y = iris.target

# Оставляем только 2 класса (для бинарной классификации)
X = X[y != 2]
y = y[y != 2]

# Обучение SVM
model = svm.SVC(kernel='linear', C=1.0)
model.fit(X, y)

# Визуализация
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='autumn')
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Сетка для отображения разделяющей плоскости
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = model.decision_function(xy).reshape(XX.shape)

# Границы и опорные векторы
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], linestyles=['--', '-', '--'])
ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')
plt.savefig("SVM.png")
plt.close()