def logistic_function(x):
    """ Applies the logistic function to x, element-wise. """
    
    return 1 / (1 + np.exp(-x))


def x_strich(x):
    xt = np.zeros((x.shape[0],x.shape[1]+1))
    xt[:,:1] = 1
    xt[:,1:] = x
    return xt


def logistic_hypothesis(theta):
    ''' Combines given list argument in a logistic equation and returns it as a function
    
    Args:
        thetas: list of coefficients
        
    Returns:
        lambda that models a logistc function based on thetas and x
    '''
    
    return lambda x: logistic_function(x_strich(x).dot(theta))