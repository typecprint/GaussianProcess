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
    return np.sin(2.0 * np.pi * 0.2 * x) + np.sin(2.0 * np.pi * 0.1 * x)


np.random.seed(10)
n_observation = 50

func = func1d
x_train = np.random.uniform(0.0, 10.0, n_observation)
y_train = func(x_train) + np.random.normal(0, 0.3, len(x_train))
x = np.linspace(-2.0, 12.0, 100)

plt.figure(figsize=(10, 8))
plt.scatter(x_train, y_train, marker="o", color="r", label="Data Point")
plt.plot(x, func(x), label="True Distribution")
plt.legend()
# plt.savefig("./img/1d-raw-data.png")

# %%
theta = [1.0, 1.0, 0, 0]
beta = 1 / np.var(y_train)
gp = GaussianProcess(theta, beta)
gp.set_training_data(x_train, y_train)

# %%
# gaussian kernel check
# pred = np.array([gp.gp_sampling(x) for i in range(5)])
# fig, ax = plt.subplots()
# for i, p in enumerate(pred):
#     ax.plot(x, p)
#     ax.set_xlabel("x")
#     ax.set_ylabel("y")
#     ax.set_title("Gaussian Kernel")

# %%
# gaussian process
mu, var = gp.prediction(x)
plt.figure(figsize=(10, 8))
plt.plot(x, func(x), "x", color="green", label="True distribution")
plt.plot(x_train, y_train, "o", color="red", label="Observation")

plt.plot(x, mu, color="blue", label="predict(mean)")
plt.fill_between(x, mu + np.sqrt(var), mu - np.sqrt(var), alpha=0.2, color="blue")
plt.legend()
# plt.savefig("./img/1d-gpr.png")

# %%
gp.optimize()
mu, var = gp.prediction(x)
plt.figure(figsize=(10, 8))
plt.plot(x, func(x), "x", color="green", label="True distribution")
plt.plot(x_train, y_train, "o", color="red", label="Observation")
plt.plot(x, mu, color="blue", label="predict(mean)")
plt.fill_between(x, mu + np.sqrt(var), mu - np.sqrt(var), alpha=0.2, color="blue")
plt.legend()
# plt.savefig("./img/1d-gpr-optimize.png")

# %%
# GPy
kernel = GPy.kern.RBF(1, variance=1)
model = GPy.models.GPRegression(x_train[:, None], y_train[:, None], kernel=kernel)
mu, var = model.predict(x[:, None])
plt.figure(figsize=(10, 8))
plt.plot(x, func(x), "x", color="green", label="True distribution")
plt.plot(x_train, y_train, "o", color="red", label="Observation")
plt.plot(x, mu, color="blue", label="predict(mean)")
plt.fill_between(
    x,
    mu[:, 0] + np.sqrt(var[:, 0]),
    mu[:, 0] - np.sqrt(var[:, 0]),
    alpha=0.2,
    color="blue",
)
plt.legend()
# plt.savefig("./img/1d-gpy-result.png")
plt.show()

# %%
model.optimize()
mu, var = model.predict(x[:, None])
plt.figure(figsize=(10, 8))
plt.plot(x, func(x), "x", color="green", label="True distribution")
plt.plot(x_train, y_train, "o", color="red", label="Observation")
plt.plot(x, mu, color="blue", label="predict(mean)")
plt.fill_between(
    x,
    mu[:, 0] + np.sqrt(var[:, 0]),
    mu[:, 0] - np.sqrt(var[:, 0]),
    alpha=0.2,
    color="blue",
)
plt.legend()
# plt.savefig("./img/1d-gpy-result-optimize.png")
plt.show()


# %%

# %%
