import numpy as np


def sigmoid(x):
    """ Applies the logistic function to x, element-wise. """
    return 1 / (1 + np.exp(-x))


def x_strich(x):
    xt = numpy.zeros((x.shape[0], x.shape[1] + 1))
    xt[:, :1] = 1
    xt[:, 1:] = x
    return xt


def logistic_hypothesis(theta):
    """Combines given list argument in a logistic equation and returns it as a
    function
    Args:
        thetas: list of coefficients
    Returns:
        lambda that models a logistc function based on thetas and x
    """
    return lambda X: sigmoid(x_strich(X).dot(theta))


def L2_regularization_cost(X, theta, lambd):
    print(X.shape)
    return np.sum(np.square(theta)) * (lambd / (2 * X.shape[0]))


def cross_entropy_costs(h, X, y):
    """Implements cross-entropy as a function costs(theta) on given traning data
    Args:
        h: the hypothesis as function
        x: features as 2D array with shape (m_examples, n_features)
        y: ground truth labels for given features with shape (m_examples)
    Returns:
        lambda costs(theta) that models the cross-entropy for each x^i
    """
    return lambda theta: -y * np.log(h(theta)(X)) - (1 - y) * np.log(1 - h(theta)(X))


def regulated_cost(h, X, y, theta, lambd):
    return cross_entropy_costs(h, X, y) + L2_regularization_cost(X, theta, lambd)


def compute_new_theta(X, y, theta, learning_rate, hypothesis, lambda_reg=0.1):
    """Updates learnable parameters theta
    The update is done by calculating the partial derivities of
    the cost function including the linear hypothesis. The
    gradients scaled by a scalar are subtracted from the given
    theta values.
    Args:
        X: 2D numpy array of x values
        y: array of y values corresponding to x
        theta: current theta values
        learning_rate: value to scale the negative gradient
        hypothesis: the hypothesis as function
    Returns:
        theta: Updated theta_0
    """
    h = hypothesis(theta)(X)
    return theta - learning_rate * (1 / X.shape[0]) * np.sum((h - y).dot(x_strich(X)))


def mean_cross_entropy_costs(X, y, hypothesis, cost_func, lambda_reg=0.1):
    """Implements mean cross-entropy as a function J(theta) on given traning
    data
    Args:
        X: features as 2D array with shape (m_examples, n_features)
        y: ground truth labels for given features with shape (m_examples)
        hypothesis: the hypothesis as function
        cost_func: cost function
    Returns:
        lambda J(theta) that models the mean cross-entropy
    """
    return lambda theta: np.mean(cost_func(hypothesis, X, y)(theta))


def gradient_descent(X, y, theta, learning_rate, num_iters, lambda_reg=0.1):
    """Minimize theta values of a logistic model based on cross-entropy cost function
    Args:
        X: 2D numpy array of x values
        y: array of y values corresponding to x
        theta: current theta values
        learning_rate: value to scale the negative gradient
        num_iters: number of iterations updating thetas
        lambda_reg: regularization strength
    Returns:
        history_cost: cost after each iteration
        history_theta: Updated theta values after each iteration
    """
    thetas = np.zeros((num_iters, len(theta)))
    cost = np.zeros(num_iters)
    J = mean_cross_entropy_costs(X, y, logistic_hypothesis, cross_entropy_costs)
    tmp = theta
    for i in range(num_iters):
        thetas[i] = tmp
        tmp = compute_new_theta(X, y, thetas[i], learning_rate, logistic_hypothesis)
        cost[i] = J(thetas[i])
        if i % 250 == 0:
            print(f"{i} Iteration -- {thetas[i]}")
            print(f"Cost: {cost[i]}\n")
    return cost, thetas