# from sklearn.utils.estimator_checks import
import logging
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class NumberDaysNotLogedClassifier(BaseEstimator, ClassifierMixin):
    """Predicts all the students that hasn't accessed the VLE so far as at-risk (NotSubmit)"""

    def __init__(self, last_login_column_name='last_login_rel', last_logged_days_threshold=14, interpolate=False):
        self._MAX_VALUE = 999
        self.last_login_column_name = last_login_column_name
        self.last_login_column_index = 0
        self.last_logged_days_threshold=last_logged_days_threshold
        self.interpolate = interpolate
        logging.debug("Instatiang")
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
        last_logged = X[:, self.last_login_column_index]
        return np.where((last_logged > self.last_logged_days_threshold) | np.isnan(last_logged), 1.0, 0.0)

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
        if not hasattr(self, "last_login_column_index"):
            raise ValueError("NumberDaysNotLogedClassifier not fitted.")

        last_logged = X[:, self.last_login_column_index]
        # mask = (first_access >= self._MAX_VALUE) | np.isnan(first_access)
        out = np.zeros((len(last_logged), 2), dtype=np.float64)
        mask_one = (last_logged > self.last_logged_days_threshold) | np.isnan(last_logged)
        mask_half = (last_logged > self.last_logged_days_threshold / 2) & (last_logged <= self.last_logged_days_threshold )

        mask_025 = (last_logged > self.last_logged_days_threshold  * 0.25) & (last_logged <= self.last_logged_days_threshold * 0.5 )
        mask_075 = (last_logged > self.last_logged_days_threshold * 0.75) & (last_logged <= self.last_logged_days_threshold )


        # TODO: set the probability proportionally to the days the student hasn't logged in to the VLE
        out[mask_one,  1] = 1.0
        out[~mask_one, 0] = 1.0

        if self.interpolate:
            out[~mask_one, 1] = last_logged[~mask_one] / self.last_logged_days_threshold
            out[mask_one, 0] = last_logged[mask_one] / self.last_logged_days_threshold

        # out[mask_half,  1] = 0.5
        # out[~mask_half, 0] = 0.5
        #
        # out[mask_025,  1] = 0.25
        # out[~mask_025, 0] = 0.25
        #
        # out[mask_075,  1] = 0.75
        # out[~mask_075, 0] = 0.75


        return out

    def get_params(self, deep=True):
        return { "last_login_column_name": self.last_login_column_name}
