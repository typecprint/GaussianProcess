# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import emcee
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
import GPy
import seaborn as sns
from GPR import GaussianProcess

# %%
# create data
def func1d(x):
    return np.sin(2.0 * np.pi * 0.2 * x) + np.sin(2.0 * np.pi * 0.05 * x)


df = pd.read_csv("./advertising.csv")
x = df[["TV", "Radio"]].values
y = df["Sales"].values

x = (x - x.mean(axis=0)) / x.std(axis=0)
# x = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
y = (y - y.min()) / (y.max() - y.min())
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2)

for i, angle in enumerate(range(0, 360, 120)):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x_train[:, 0], x_train[:, 1], y_train, c="steelblue", label="Training")
    ax.scatter(x_test[:, 0], x_test[:, 1], y_test, c="crimson", label="Test")
    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")
    ax.view_init(30, angle)
    plt.legend()
    plt.savefig(f"./img/2d-raw-data_{i}.png")
    plt.show()

# %%
theta = [1.0, 1.0, 0, 0]
beta = 1 / np.var(y_train)
gp = GaussianProcess(theta, beta)
gp.set_training_data(x_train, y_train)

# %%
# gaussian kernel check
pred = np.array([gp.gp_sampling(x) for i in range(5)])

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
for i, p in enumerate(pred):
    ax.scatter(x[:, 0], x[:, 1], p, marker=".")
    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")

    ax.view_init(30, angle)
plt.show()

# %%
# gaussian process
# mu, var = gp.prediction(x)

# %%
mu, var = gp.prediction(x)

# %%
X = np.arange(-1.5, 1.5, 0.1)
Y = np.arange(-1.5, 1.5, 0.1)
X, Y = np.meshgrid(X, Y)
x = []
for _xx, _yy in zip(X, Y):
    for _x, _y in zip(_xx, _yy):
        x.append(np.array([_x, _y]))
x = np.array(x)
mu, std = gp.prediction(x)

for i, angle in enumerate(range(0, 360, 90)):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        x_train[:, 0],
        x_train[:, 1],
        y_train,
        marker="o",
        color="r",
        label="Training",
    )
    ax.scatter(x_test[:, 0], x_test[:, 1], y_test, marker="x", color="g", label="Test")
    ax.plot_surface(X, Y, mu.reshape(X.shape[0], -1), alpha=0.3)
    ax.view_init(0, angle)
    plt.legend()
    plt.savefig(f"./img/2d-gpr-{i}")
    plt.show()

# %%
gp.optimize()

# %%
