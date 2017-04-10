import numpy as np
import matplotlib.pyplot as plt
from logistic import LogisticRegression

# read data
X = np.loadtxt('logistic_x.txt')
y = np.loadtxt('logistic_y.txt')

# build model
lr = LogisticRegression()
lr.fit(X, y)
y_ = lr.predict(X)

# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = 0.1  # step_size
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
data = np.vstack((xx.ravel(), yy.ravel())).T
labels = lr.predict(data)

# plot
fig, ax = plt.subplots()
ax.scatter(data[:, 0], data[:, 1],
           c=np.where(labels == 1, 'green', 'red'), alpha=0.01)
plt.title('Decision Boundary of Logistic Regression')
ax.scatter(X[y == 1, 0], X[y == 1, 1], c='green',
           label='1', marker='x', alpha=0.7)
ax.scatter(X[y == -1, 0], X[y == -1, 1], c='red',
           label='-1', marker='x', alpha=0.7)

plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title('Decision Boundary of Logistic Regression')
plt.legend()

fig.savefig('logistic_reg_decision_boundary.png')
