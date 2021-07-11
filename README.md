# Gaussian Process

# Description

![gif](./image/animation.gif)

A Python implementation of Gaussian process regression.

- Supporting multi-dimensional Gaussian process regression.
- Kernel function is a Gaussian kernel with a constant term and a linear term.

# Example

## 1 dimension data

### Input Data

Let's create a simple sine curve with some noise on it and perform a Gaussian process regression.

![input](./image/1d-raw-data.png)

### Regression Result

This is the result of fixing the kernel function with arbitrary parameters and performing a Gaussian process regression. The red dots represent the observed points, the green color is GroundTruth, the dark blue line is the mean of the regression, and the light blue color is plus or minus 1 sigma.

![result](./image/1d-gpr.png)

### Regression Result(Using Optimized Parameters.)

After hyperparameter optimization, the graph looks like this. This result is consistent with the Gpy library, confirming that my implementation is correct.

![result_opt](./image/1d-gpr-optimize.png)

## 2 dimension data

Here we use the famous advertising dataset as an example.

### Input Data

It can be used for two-dimensional data in exactly the same way. The blue points are the observed points and the red points are the unobserved test data.

![input](./image/2d-raw-data_1.png)

### Regression Result

If we draw a regression plane, it will look like the following

![result](./image/2d-gpr-1.png)
