# from sklearn.utils.estimator_checks import
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class NeverActiveStudentClassifier(BaseEstimator, ClassifierMixin):
    """Predicts all the students that hasn't accessed the VLE so far as at-risk (NotSubmit)"""

    def __init__(self, never_logged_column_name='never_logged'):
        self._MAX_VALUE = 999
        self.never_logged_column_name = never_logged_column_name
        self.never_logged_column_index = 0
        pass

    def fit(self, X, y):
        return self

    def predict(self, X):
        """
        Returns 1 if the student hasn't accessed the VLE or 0. The 'not accessed' flag is indicated
        by value greater or eq. than maximum expected value - MAX_VALUE or np.nan.
        :param X:
        :return:
        """
        never_logged = X[:, self.never_logged_column_index]
        return np.where((never_logged > 0) | np.isnan(never_logged), 1.0, 0.0)

    def predict_proba(self, X):
        """
            Return probability estimates for the test vectors X.
            Parameters
            ----------
            X : {array-like, sparse matrix}, shape = [n_samples, n_features]
                Input vectors, where n_samples is the number of samples
                and n_features is the number of features.
            Returns
            -------
            P : array-like or list of array-lke of shape = [n_samples, n_classes]
                Returns the probability of the sample for each class in
                the model, where classes are ordered arithmetically, for each
                output.
            """
        if not hasattr(self, "never_logged_column_index"):
            raise ValueError("NeverActiveStudentClassifier not fitted.")

        never_logged = X[:, self.never_logged_column_index]
        # mask = (first_access >= self._MAX_VALUE) | np.isnan(first_access)
        mask = (never_logged > 0) | np.isnan(never_logged)
        out = np.zeros((len(never_logged), 2), dtype=np.float64)

        out[mask,  1] = 1.0
        out[~mask, 0] = 1.0

        return out

    def get_params(self, deep=True):
        return {"never_logged_column_name": self.never_logged_column_name
                }
