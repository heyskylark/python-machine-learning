import numpy as np

class LogisticRegressionGD(object):
  """
  Logistics Regression

  Parameters
  ----------
  eta : float
    Learning rate (between 0.0 and 1.0)
  n_iter : int
    Passes over the training dataset
  random_state : int
    Random number generator seed for random weight initilization

  Attributes
  ----------
  w_: 1d-array
    Weights after fitting
  cost_ : list
  """
  def __init__(self, eta=0.05, n_iter=100, random_state=1):
    self.eta = eta
    self.n_iter = n_iter
    self.random_state = random_state

  def fit(self, X, y):
    """
    Fit training data.  

    Parameters
    ----------
    X : {array-like}, shape = [n_samples, n_features]
      Training vectors, where n_samples is the number of samples and n_features is the number of features
    Y : {array-like}, shape = [n_samples]
      Target values

    Returns
    -------
    self : object
    """
    rgen = np.random.RandomState(self.random_state)
    self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])

    self.cost_ = []

    for i in range(self.n_iter):
      net_input = self.net_input(X)
      output = self.activation(net_input)
      errors = (y - output)
      self.w_[1:] += self.eta * X.T.dot(errors)
      self.w_[0] += self.eta * errors.sum()
      # Note: we are now using logistic cost function instead of the squared error
      cost = (-y.dot(np.log(output)))-((1 - y).dot(np.log(1 - output)))
      self.cost_.append(cost)

    return self
    
  def net_input(self, X):
    return np.dot(X, self.w_[1:]) + self.w_[0]

  def activation(self, z):
    """Sigmoid activation with a clip of (-250, 250)"""
    return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

  def predict(self, X):
    """
    - Net-input prediction
    - Sigmoid activation
    - Step function conversion to 1 or 0
    """
    return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
