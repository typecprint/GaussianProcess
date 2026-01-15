import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve


class GaussianProcessEfficient:
    """
    Implementation focusing on computational efficiency and numerical stability.
    - Vectorized matrix operations for speed.
    - Cholesky decomposition to avoid direct matrix inversion.
    - ARD (Automatic Relevance Determination) kernel for improved flexibility.
    - Optimized gradient calculation for hyperparameter tuning.
    """

    def __init__(self, theta, beta, x_train=None, y_train=None):
        self.theta = np.array(theta, dtype=np.float64)
        self.beta = float(beta)
        self.jitter = 1e-8
        self.dim = None
        self.param_bound = None

        self.x_train = None
        self.y_train = None
        if x_train is not None and y_train is not None:
            self.set_training_data(x_train, y_train)

    def __data_checker(self, x):
        if x is None:
            return None
        return np.atleast_2d(x)

    def create_kernel_mtx(self, x1, x2, include_noise=False):
        """Constructs the kernel matrix using a combined ARD RBF, Linear, and Constant kernel."""
        if self.dim is None:
            raise ValueError(
                "Training data must be set before creating a kernel matrix."
            )

        x1 = self.__data_checker(x1)
        x2 = self.__data_checker(x2)

        # ARD RBF Kernel part
        length_scales = self.theta[1 : 1 + self.dim]
        # Element-wise multiplication of each sample with length-scale params
        x1_scaled = x1 * length_scales
        x2_scaled = x2 * length_scales

        # Squared Euclidean distance on scaled data
        dist_sq = (
            np.sum(x1_scaled**2, axis=1, keepdims=True)
            + np.sum(x2_scaled**2, axis=1)
            - 2 * np.dot(x1_scaled, x2_scaled.T)
        )

        K = self.theta[0] * np.exp(-0.5 * dist_sq)

        # Add Linear and Constant (Bias) kernels
        K += self.theta[1 + self.dim]  # Constant kernel
        K += self.theta[2 + self.dim] * np.dot(x1, x2.T)  # Linear kernel

        if include_noise:
            K += (1.0 / self.beta + self.jitter) * np.eye(x1.shape[0])

        return K

    def prediction(self, x_test):
        x_test = self.__data_checker(x_test)
        K = self.create_kernel_mtx(self.x_train, self.x_train, include_noise=True)

        L, lower = cho_factor(K)
        k_star = self.create_kernel_mtx(self.x_train, x_test)

        alpha = cho_solve((L, lower), self.y_train)
        mu = np.dot(k_star.T, alpha)

        v = cho_solve((L, lower), k_star)

        # Diagonal of k(x_test, x_test)
        # RBF variance + Constant variance + Linear variance
        k_star_star_diag = (
            self.theta[0]
            + self.theta[1 + self.dim]
            + self.theta[2 + self.dim] * np.sum(x_test**2, axis=1)
        )
        var = k_star_star_diag - np.sum(k_star * v, axis=0)

        return mu, var

    def optimize(self):
        if self.dim is None:
            raise ValueError("Training data must be set before optimizing.")

        def objective(params):
            self.beta = params[0]
            self.theta = params[1:]

            K = self.create_kernel_mtx(self.x_train, self.x_train, include_noise=True)
            try:
                L, lower = cho_factor(K)
            except np.linalg.LinAlgError:
                return 1e10, np.zeros_like(params)

            alpha = cho_solve((L, lower), self.y_train)
            K_inv = cho_solve((L, lower), np.eye(len(K)))
            W = np.outer(alpha, alpha) - K_inv

            # Log-Marginal Likelihood
            lml = (
                -0.5 * np.dot(self.y_train, alpha)
                - np.sum(np.log(np.diag(L)))
                - 0.5 * len(self.y_train) * np.log(2 * np.pi)
            )

            # --- Gradient Calculation ---
            grad = np.zeros_like(params)

            # Gradient w.r.t. beta (noise)
            dk_dbeta = -(1.0 / (self.beta**2)) * np.eye(len(K))
            grad[0] = 0.5 * np.sum(W * dk_dbeta)

            # --- Gradients w.r.t. theta ---
            length_scales = self.theta[1 : 1 + self.dim]
            x_train_scaled = self.x_train * length_scales
            dist_sq = (
                np.sum(x_train_scaled**2, axis=1, keepdims=True)
                + np.sum(x_train_scaled**2, axis=1)
                - 2 * np.dot(x_train_scaled, x_train_scaled.T)
            )
            rbf_part = np.exp(-0.5 * dist_sq)

            # dK/d(theta_0) - RBF variance
            dk_dtheta0 = rbf_part
            grad[1] = 0.5 * np.sum(W * dk_dtheta0)

            # dK/d(theta_l) - ARD length-scales
            for i in range(self.dim):
                # Distance calculation for a single dimension
                dim_dist_sq = (
                    np.sum(self.x_train[:, i : i + 1] ** 2, axis=1, keepdims=True)
                    + np.sum(self.x_train[:, i : i + 1] ** 2, axis=1)
                    - 2
                    * np.dot(self.x_train[:, i : i + 1], self.x_train[:, i : i + 1].T)
                )
                dk_dl = (
                    self.theta[0]
                    * rbf_part
                    * (-0.5 * dim_dist_sq)
                    * (2 * length_scales[i])
                )
                grad[2 + i] = 0.5 * np.sum(W * dk_dl)

            # dK/d(theta_bias)
            dk_dconst = np.ones_like(K)
            grad[2 + self.dim] = 0.5 * np.sum(W * dk_dconst)

            # dK/d(theta_linear)
            dk_dlinear = np.dot(self.x_train, self.x_train.T)
            grad[3 + self.dim] = 0.5 * np.sum(W * dk_dlinear)

            return -lml, -grad

        initial_params = np.append(self.beta, self.theta)
        res = minimize(
            objective,
            initial_params,
            jac=True,
            bounds=self.param_bound,
            method="L-BFGS-B",
        )

        if res.success:
            self.beta = res.x[0]
            self.theta = res.x[1:]
        return res

    def set_training_data(self, x_train, y_train):
        self.x_train = self.__data_checker(x_train)
        self.y_train = y_train.flatten()
        self.dim = self.x_train.shape[1]

        # Verbose calculation for the number of parameters to ensure correctness.
        # The total number of parameters to optimize includes:
        # 1 for beta (noise precision)
        # 1 for RBF variance (theta_0)
        # 'self.dim' for ARD length-scales (one for each dimension)
        # 1 for the constant bias kernel
        # 1 for the linear kernel
        num_rbf_var_params = 1
        num_length_scale_params = self.dim
        num_bias_params = 1
        num_linear_params = 1

        num_theta_params = (
            num_rbf_var_params
            + num_length_scale_params
            + num_bias_params
            + num_linear_params
        )

        num_total_params = 1 + num_theta_params  # Add 1 for beta
        self.param_bound = [(1e-6, None)] * num_total_params

    def gp_sampling(self, x, n_samples=1):
        x = self.__data_checker(x)
        K = self.create_kernel_mtx(x, x) + self.jitter * np.eye(len(x))
        L = np.linalg.cholesky(K)
        samples = np.dot(L, np.random.normal(size=(len(x), n_samples)))
        return samples.T if n_samples > 1 else samples.flatten()
