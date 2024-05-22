import numpy as np

class AdalineSGD(object):
  """
  ADAptive LInear NEuron classifier

  Parameters
  ----------
  eta : float
    Learning rate (between 0.0 and 1.0)
  n_iter : int
    Passes over the training dataset
  shuffle : bool (default: True)
    Shuffles training data every epoch if True to prevent cycles
  random_state : int
    Random number generator seed for random weight initialization

  Attributes
  ----------
  w_ : 1d-array
    Weights after fitting
  cost_ : list
  """

  def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
    self.eta = eta
    self.n_iter = n_iter
    self.random_state = random_state
    self.shuffle = shuffle

  def fit(self, X, Y):
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
    self._initialize_weights(X.shape[1])
    self.cost_ = []
    for i in range(self.n_iter):
      if self.shuffle:
        X, Y = self._shuffle(X, Y)

      cost = []
      for xi, target in zip(X, Y):
        cost.append(self._update_weights(xi, target))

      avg_cost = sum(cost) / len(Y)
      self.cost_.append(avg_cost)

    return self
  
  def partial_fit(self, X, Y):
    """Fit training data without reinitializing the weights"""
    if not hasattr(self, 'w_'):
      self._initialize_weights(X.shape[1])

    if Y.ravel().shape[0] > 1:
      for xi, target in zip(X, Y):
        self._update_weights(xi, target)
    else:
      self._update_weights(X, Y)

    return self
  
  def _shuffle(self, X, Y):
    """Shuffle training data"""
    r = self.rgen.permutation(len(Y))
    return X[r], Y[r]
  
  def _initialize_weights(self, m):
    """Initialize weights to small random numbers"""
    self.rgen = np.random.RandomState(self.random_state)
    self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1 + m)
    self.w_initialized = True

  def _update_weights(self, xi, target):
    """Apply Adaline learning rule to update the weights"""
    output = self.activation(self.net_input(xi))
    error = (target - output)
    self.w_[1:] += self.eta * xi.dot(error)
    self.w_[0] += self.eta * error
    cost = error**2 / 2.0
    return cost
  
  def net_input(self, X):
    """Calculate net input"""
    return np.dot(X, self.w_[1:]) + self.w_[0]
  
  def activation(self, X):
    """Compute linear activation"""
    return X
  
  def predict(self, X):
    """Return class label after unit step"""
    return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)