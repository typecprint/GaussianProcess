import numpy as np
from scipy.optimize import minimize


class GaussianProcess:
    def __init__(self, theta, beta, x_train=None, y_train=None, param_bound=None):
        self.theta = theta
        self.beta = beta
        if param_bound is None:
            self.param_bound = np.array(
                [[0.0, None], [0.0, None], [0.0, None], [0.0, None], [0.0, None]]
            )
        self.x_train = x_train
        self.y_train = y_train
        if self.x_train is not None:
            self.dim = 2 if len(x_train.shape) == 2 else 1
        self.kernel = self.__gaussian_kernel

    def create_kernel_mtx(self, x1, x2):
        x1_ = self.__data_checker(x1)
        x2_ = self.__data_checker(x2)
        K = np.array(
            [self.kernel(_x1, _x2, np.all(_x1 == _x2)) for _x1 in x1_ for _x2 in x2_]
        )
        K = K.reshape(x1_.shape[0], x2_.shape[0])
        return K

    def gp_sampling(self, x):
        K = self.create_kernel_mtx(x, x)
        L = np.linalg.cholesky(K)
        _x = np.random.normal(0, 1, len(x))
        return L @ _x

    def prediction(self, x):
        mu = []
        var = []

        Cn = self.create_kernel_mtx(self.x_train, self.x_train)
        Cn_inv = np.linalg.inv(Cn)

        k = self.create_kernel_mtx(self.x_train, x)

        # Actuary only use diagonal matrix. It can be made faster.
        c = self.create_kernel_mtx(x, x)
        mu = k.T @ Cn_inv @ self.y_train
        all_var = c - (k.T @ Cn_inv @ k)
        var = np.diag(all_var)
        return mu, var

    # def prediction__(self, x):
    #     mu = []
    #     var = []
    #     if self.dim == 2:
    #         x_shape = x[np.newaxis, :, :]
    #     else:
    #         x_shape = x

    #     for _x in x_shape:
    #         self.Cn = self.create_kernel_mtx(self.x_train, self.x_train)
    #         self.Cn_inv = np.linalg.inv(self.Cn)
    #         k = self.create_kernel_mtx(self.x_train, _x)
    #         c = self.create_kernel_mtx(_x, _x)
    #         _mu = k.T @ self.Cn_inv @ self.y_train
    #         _var = c - (k.T @ self.Cn_inv @ k)
    #         mu.append(_mu)
    #         var.append(_var)
    #     return np.array(mu), np.array(var)

    def set_training_data(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.dim = 2 if len(x_train.shape) == 2 else 1

    def set_parameters(self, _beta, _theta):
        self.beta = _beta
        self.theta = _theta

    def optimize(self):
        def __objective(_beta, _theta):
            self.set_parameters(_beta, _theta)
            Cn = self.create_kernel_mtx(self.x_train, self.x_train)
            Cn_inv = np.linalg.inv(Cn)
            partial_C = self.__partial_diff(self.x_train, self.x_train)
            L = np.array(
                -0.5 * np.linalg.slogdet(Cn)[1]
                - 0.5 * self.y_train.T @ Cn_inv @ self.y_train
                - 0.5 * len(self.x_train) * np.log(2 * np.pi)
            )

            dbeta = np.array(
                -0.5 * np.trace(Cn_inv) / (_beta ** 2)
                + 0.5 / (_beta ** 2) * (self.y_train.T @ Cn_inv @ Cn_inv @ self.y_train)
            )

            dtheta = np.array(
                [
                    0.5 * np.trace(Cn_inv @ p_c)
                    - 0.5 * self.y_train.T @ Cn_inv @ p_c @ Cn_inv @ self.y_train
                    for p_c in partial_C.T
                ]
            )

            dL = np.append(dbeta, dtheta)
            return -L, dL

        old_beta = self.beta
        old_theta = self.theta
        param_bounds = self.param_bound
        result = minimize(
            x0=np.append(self.beta, self.theta),
            fun=lambda param: __objective(_beta=param[0], _theta=param[1:]),
            jac=True,
            bounds=param_bounds,
        )
        print("Optimize:: ", result.success)
        if result.success:
            self.set_parameters(result.x[0], result.x[1:])
        else:
            self.set_parameters(old_beta, old_theta)

    @staticmethod
    def __data_checker(x):
        if x.shape == ():
            x = x.reshape(1, 1)
        elif len(x.shape) == 1:
            x = x.reshape(-1, 1)
        return x

    def __gaussian_kernel(self, x, x_prime, delta):
        return np.squeeze(
            self.theta[0]
            * np.exp(-self.theta[1] / 2 * np.linalg.norm(x - x_prime) ** 2)
            + self.theta[2]
            + self.theta[3] * x.T @ x_prime
            + 1 / self.beta * delta
        )

    def __partial_diff(self, x1, x2):
        def f_dtheta0(x, x_prime):
            return np.exp(-self.theta[1] / 2 * np.linalg.norm(x - x_prime) ** 2)

        def f_dtheta1(x, x_prime):
            _d = np.linalg.norm(x - x_prime) ** 2
            return -0.5 * _d * self.theta[0] * (np.exp(-0.5 * self.theta[1] * _d))

        def f_dtheta2(x, x_prime):
            return 1

        def f_dtheta3(x, x_prime):
            return x.T @ x_prime

        x1_ = self.__data_checker(x1)
        x2_ = self.__data_checker(x2)
        dtheta = np.array(
            [
                [
                    f_dtheta0(_x1, _x2),
                    f_dtheta1(_x1, _x2),
                    f_dtheta2(_x1, _x2),
                    f_dtheta3(_x1, _x2),
                ]
                for _x1 in x1_
                for _x2 in x2_
            ],
            dtype=np.float32,
        )
        return dtheta.reshape(x1_.shape[0], x2_.shape[0], 4)
