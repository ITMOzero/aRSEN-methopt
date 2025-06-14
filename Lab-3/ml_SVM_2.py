import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

X = np.array([[1, 2], [2, 3], [3, 3], [6, 5], [7, 8], [8, 8]])
y = np.array([1, 1, 1, -1, -1, -1])

model = svm.SVC(kernel='linear', C=1.0)
model.fit(X, y)

w = model.coef_[0]
b = model.intercept_[0]
print(f"Гиперплоскость: {w[0]:.2f}x1 + {w[1]:.2f}x2 + {b:.2f} = 0")

print("Опорные векторы:", model.support_vectors_)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='autumn')
ax = plt.gca()

xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = model.decision_function(xy).reshape(XX.shape)

ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], linestyles=['--', '-', '--'])
ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100, facecolors='none', edgecolors='k')
plt.savefig("SVM_2.png")
plt.close()
