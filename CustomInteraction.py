import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class CustomInteraction(TransformerMixin, BaseEstimator):
    """Make user defined interaction between columns."""

    def __init__(self, *, interaction_columns=tuple()):
        self.interaction_columns = interaction_columns

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.
        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features.
            - If `input_features is None`, then `feature_names_in_` is
              used as feature names in. If `feature_names_in_` is not defined,
              then names are generated: `[x0, x1, ..., x(n_features_in_)]`.
            - If `input_features` is an array-like, then `input_features` must
              match `feature_names_in_` if `feature_names_in_` is defined.
        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        feature_names = []
        for index, name in enumerate(input_features):
            for int_col in self.interaction_columns:
                if index == int_col:
                    feature_names.append(f'{name}**2')
                else:
                    feature_names.append(f'{name}*{input_features[int_col]}')
        return np.asarray(feature_names, dtype=object)

    def fit(self, X, y=None):
        """
        Compute number of output features.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data.
        y : Ignored
            Not used, present here for API consistency by convention.
        Returns
        -------
        self : object
            Fitted transformer.
        """
        self._n_output_features = len(self.interaction_columns)*(X.shape[1]+1)
        return self

    def transform(self, X):
        """Transform data to polynomial features.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data to transform, row by row.
            Prefer CSR over CSC for sparse input (for speed), but CSC is
            required if the degree is 4 or higher. If the degree is less than
            4 and the input format is CSC, it will be converted to CSR, have
            its polynomial features generated, then converted back to CSC.
            If the degree is 2 or 3, the method described in "Leveraging
            Sparsity to Speed Up Polynomial Feature Expansions of CSR Matrices
            Using K-Simplex Numbers" by Andrew Nystrom and John Hughes is
            used, which is much faster than the method used on CSC input. For
            this reason, a CSC input will be converted to CSR, and the output
            will be converted back to CSC prior to being returned, hence the
            preference of CSR.
        Returns
        -------
        XP : {ndarray, sparse matrix} of shape (n_samples, NP)
            The matrix of features, where `NP` is the number of polynomial
            features generated from the combination of inputs. If a sparse
            matrix is provided, it will be converted into a sparse
            `csr_matrix`.
        """
        n_samples = X.shape[0]
        n_int_cols = len(self.interaction_columns)
        XP = np.empty(
            shape=(n_samples, self._n_output_features), dtype=X.dtype, order='C',
        )

        XP[:, 0:len(self.interaction_columns)] = (
            X[:, self.interaction_columns])
        print(X)
        print(XP)
        for index in range(X.shape[1]):
            for int_col_index, int_col in enumerate(self.interaction_columns):
                print(f'{n_int_cols+index*int_col_index}')
                XP[:, n_int_cols+index*(int_col_index+1)] = (
                    X[:, index] * X[:, int_col])
                #print(f'{X[:, index]} * {X[:, int_col]} = {X[:, index]*X[:, int_col]}')
        print(f'done with transform {XP.shape} input {X.shape}')
        return XP
