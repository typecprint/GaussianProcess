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


x = np.linspace(-2.0, 12.0, 100)
func = func1d
np.random.seed(4)

x_train = np.array(np.random.uniform(-1, 11, 1)[0], dtype=np.float32)
y_train = np.array(func(x_train) + np.random.normal(0, 0.3, 1), dtype=np.float32)

for i in range(50):
    print(f"Observation Num: {i+1}")
    add_x = np.array(np.random.uniform(-1, 11, 1)[0], dtype=np.float32)
    add_y = np.array(func(add_x) + np.random.normal(0, 0.3, 1), dtype=np.float32)

    x_train = np.append(x_train, add_x)
    y_train = np.append(y_train, add_y)

    theta = np.array([1.0, 1.0, 0, 0])
    beta = 1 / np.var(y_train)
    gp = GaussianProcess(theta, beta)
    gp.set_training_data(x_train, y_train)

    gp.optimize()
    mu, var = gp.prediction(x)
    plt.figure(figsize=(10, 8))
    plt.plot(x, func(x), "x", color="green", label="True distribution")
    plt.plot(x_train, y_train, "o", color="red", label="Observation")
    plt.plot(x, mu, color="blue", label="predict(mean)")
    plt.fill_between(x, mu + np.sqrt(var), mu - np.sqrt(var), alpha=0.2, color="blue")
    plt.ylim([-2.5, 2.5])
    plt.xlim([-2, 12])
    plt.legend()
    plt.title("ObservationNum: %02d" % (i + 2))
    plt.savefig("./gif/gif_%04d.png" % i)
    # plt.show()
    plt.close()

# %%
