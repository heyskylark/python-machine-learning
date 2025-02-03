from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import numpy as np

class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
  """ A majority vote ensemble classifier
  Parameters:
  ----------
  classifiers : array-like, shape = [n_classifiers]
    Different classifiers for the ensemble
  vote : str, {'classlabel', 'probability'}
    Default: 'classlabel'
    If 'classlabel' the prediction is based on
    the argmax of class labels. Else if
    'probability', the argmax of the sum of
    probabilities is used to predict the class label
    (recommended for calibrated classifiers).
  weights : array-like, shape = [n_classifiers]
    Optional, default: None
    If a list of 'int' or 'float' values are
    provided, the classifiers are weighted by
    importance; Uses uniform weights if 'weights=None'.
  """

  def __init__(self, classifiers, vote='classlabel', weights=None):
    self.classifiers = classifiers
    self.named_classifiers = {key: value for
                              key, value in
                              _name_estimators(classifiers)}
    self.vote = vote
    self.weights = weights

  def fit(self, X, y):
    """
    Fit the classifiers.

    Parameters:
    ----------
    X : {array-like, sparse matrix},
        shape = [n_samples, n_features]
        Matrix of training samples.
    y : array-like,
        shape = [n_samples]
        Vector of target class labels.
    """
    # Use LabelEncoder to ensure class labels start with 0
    # which is important for np.argmax
    # call in self.predict
    self.lablenc_ = LabelEncoder()
    self.lablenc_.fit(y)
    self.classes_ = self.lablenc_.classes_
    self.classifiers_ = []
    for clf in self.classifiers:
      fitted_clf = clone(clf).fit(X, self.lablenc_.transform(y))
      self.classifiers_.append(fitted_clf)
    return self
  
  def predict(self, X):
    """
    Predict class labels for X.

    Parameters:
    ----------
    X : {array-like, sparse matrix},
        shape = [n_samples, n_features]
        Matrix of training samples.

    Returns:
    -------
    maj_vote : array-like, shape = [n_samples]
        Predicted class labels.
    """
    if self.vote == 'probability':
      maj_vote = np.argmax(self.predict_proba(X), axis=1)
    else: # 'classlabel' vote
      # Collect results from clf.predict calls
      predictions = np.asarray([clf.predict(X) for clf in self.classifiers_]).T
      maj_vote = np.apply_along_axis(lambda x:
                                    np.argmax(np.bincount(x, weights=self.weights)),
                                    axis=1,
                                    arr=predictions)
    maj_vote = self.lablenc_.inverse_transform(maj_vote)
    return maj_vote
  
  def predict_proba(self, X):
    """
    Predict class probabilities for X.

    Parameters:
    ----------
    X : {array-like, sparse matrix},
        shape = [n_samples, n_features]
        Training vectors, where
        n_examples is the number of examples and
        n_features is the number of features.

    Returns:
    -------
    avg_probas : array-like, shape = [n_samples, n_classes]
        Weighted average probability for each class per sample.
    """
    probas = np.asarray([clf.predict_proba(X) for clf in self.classifiers_])
    avg_probas = np.average(probas, axis=0, weights=self.weights)
    return avg_probas
  
  def get_params(self, deep=True):
    """ Get classifier parameter names for GridSearch"""
    if not deep:
      return super(MajorityVoteClassifier, self).get_params(deep=False)
    else:
      out = self.named_classifiers.copy()
      for name, step in self.named_classifiers.items():
        for key, value in step.get_params(deep=True).items():
          out['%s__%s' % (name, key)] = value
    return out