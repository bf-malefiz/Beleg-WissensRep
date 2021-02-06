import numpy as np
import matplotlib.pyplot as plt


def sigmoid(X):
    """ Applies the logistic function to x, element-wise. """
    return 1 / (1 + np.exp(-X))


def x_strich(X):

    return np.column_stack((np.ones(len(X)), X))


def feature_scaling(X):
    x_mean = np.mean(X)
    x_std = np.std(X)

    return (X - x_mean) / x_std


def logistic_hypothesis(theta):
    """Combines given list argument in a logistic equation and returns it as a
    function
    Args:
        thetas: list of coefficients
    Returns:
        lambda that models a logistc function based on thetas and x
    """
    return lambda X: sigmoid(np.dot(x_strich(X), theta))


# def regulated_cost(X, y, theta, lambda_reg):
#
#    return cross_entropy(X, y)(theta) + L2_regularization_cost(X, theta, lambda_reg)


# def cross_entropy(X, y):
#    """
#    Computes the cross-entropy for a single logit value and a given target class.
#    Parameters
#    ----------
#    X : float64 or float32
#    The logit
#    y : int
#    The target class
#    Returns
#    -------
#    floatX
#    The cross entropy value (negative log-likelihood)
#    """
#
#    def cost(theta):
#        z = x_strich(X).dot(theta)
#        mu = np.max([np.zeros(X.shape[0]), -z], axis=0)
#        r1 = y * (mu + np.log(np.exp(-mu) + np.exp(-z - mu)))
#        mu = np.max([np.zeros(X.shape[0]), z], axis=0)
#        r2 = (1 - y) * (mu + np.log(np.exp(-mu) + np.exp(z - mu)))
#        return r1 + r2
#
#    return cost


def cross_entropy(X, y):
    """Implements cross-entropy as a function costs(theta) on given traning data
    Args:
        h: the hypothesis as function
        x: features as 2D array with shape (m_examples, n_features)
        y: ground truth labels for given features with shape (m_examples)
    Returns:
        lambda costs(theta) that models the cross-entropy for each x^i
    """
    return lambda theta: -y * np.log(logistic_hypothesis(theta)(X) + 1e-9) - (
        1 - y
    ) * np.log(1 - logistic_hypothesis(theta)(X) + 1e-9)


def compute_new_theta(X, y, theta, learning_rate, lambda_reg):
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

    thetas = np.zeros(len(theta))
    thetas = theta * (1 - learning_rate * (lambda_reg / len(X))) - (
        learning_rate / len(X)
    ) * np.sum((logistic_hypothesis(theta)(X) - y) * x_strich(X).T, axis=1)

    return thetas


def L2_regularization_cost(X, theta, lambda_reg):
    return np.sum(theta ** 2) * (lambda_reg / (2 * len(X)))


def gradient_descent(X, y, theta, learning_rate, num_iters, lambda_reg):
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
    thetas = [theta]
    cost = np.zeros(num_iters)

    J = mean_cross_entropy_costs(X, y, cross_entropy, lambda_reg)
    cost[0] = J(thetas[0])
    for i in range(1, num_iters):
        thetas.append(compute_new_theta(X, y, thetas[i - 1], learning_rate, lambda_reg))
        cost[i] = J(thetas[i])
    return cost, thetas


def mean_cross_entropy_costs(X, y, cost_func, lambda_reg):
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
    return lambda theta: np.mean(cost_func(X, y)(theta)) + L2_regularization_cost(
        X, theta, lambda_reg
    )


def plot_progress(fig, costs, learning_rate, lambda_reg):
    """Plots the costs over the iterations

    Args:
        costs: history of costs
    """
    plt.subplot(211)
    plt.plot(
        np.arange(len(costs)),
        costs,
        alpha=0.8,
        label="LR: " + str(learning_rate) + " __ Lambda: " + str(lambda_reg),
    )
    plt.ylim(0, 1)
    plt.legend(
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc="lower left",
        ncol=4,
        mode="expand",
        borderaxespad=0.0,
    )
