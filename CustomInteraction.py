import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class PolynomialFeatures(TransformerMixin, BaseEstimator):
    """Generate polynomial and interaction features.
    Generate a new feature matrix consisting of all polynomial combinations
    of the features with degree less than or equal to the specified degree.
    For example, if an input sample is two dimensional and of the form
    [a, b], the degree-2 polynomial features are [1, a, b, a^2, ab, b^2].
    Read more in the :ref:`User Guide <polynomial_features>`.
    Parameters
    ----------
    degree : int or tuple (min_degree, max_degree), default=2
        If a single int is given, it specifies the maximal degree of the
        polynomial features. If a tuple `(min_degree, max_degree)` is passed,
        then `min_degree` is the minimum and `max_degree` is the maximum
        polynomial degree of the generated features. Note that `min_degree=0`
        and `min_degree=1` are equivalent as outputting the degree zero term is
        determined by `include_bias`.
    interaction_only : bool, default=False
        If `True`, only interaction features are produced: features that are
        products of at most `degree` *distinct* input features, i.e. terms with
        power of 2 or higher of the same input feature are excluded:
            - included: `x[0]`, `x[1]`, `x[0] * x[1]`, etc.
            - excluded: `x[0] ** 2`, `x[0] ** 2 * x[1]`, etc.
    include_bias : bool, default=True
        If `True` (default), then include a bias column, the feature in which
        all polynomial powers are zero (i.e. a column of ones - acts as an
        intercept term in a linear model).
    order : {'C', 'F'}, default='C'
        Order of output array in the dense case. `'F'` order is faster to
        compute, but may slow down subsequent estimators.
        .. versionadded:: 0.21
    Attributes
    ----------
    powers_ : ndarray of shape (`n_output_features_`, `n_features_in_`)
        `powers_[i, j]` is the exponent of the jth input in the ith output.
    n_input_features_ : int
        The total number of input features.
        .. deprecated:: 1.0
            This attribute is deprecated in 1.0 and will be removed in 1.2.
            Refer to `n_features_in_` instead.
    n_features_in_ : int
        Number of features seen during :term:`fit`.
        .. versionadded:: 0.24
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
        .. versionadded:: 1.0
    n_output_features_ : int
        The total number of polynomial output features. The number of output
        features is computed by iterating over all suitably sized combinations
        of input features.
    See Also
    --------
    SplineTransformer : Transformer that generates univariate B-spline bases
        for features.
    Notes
    -----
    Be aware that the number of features in the output array scales
    polynomially in the number of features of the input array, and
    exponentially in the degree. High degrees can cause overfitting.
    See :ref:`examples/linear_model/plot_polynomial_interpolation.py
    <sphx_glr_auto_examples_linear_model_plot_polynomial_interpolation.py>`
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.preprocessing import PolynomialFeatures
    >>> X = np.arange(6).reshape(3, 2)
    >>> X
    array([[0, 1],
           [2, 3],
           [4, 5]])
    >>> poly = PolynomialFeatures(2)
    >>> poly.fit_transform(X)
    array([[ 1.,  0.,  1.,  0.,  0.,  1.],
           [ 1.,  2.,  3.,  4.,  6.,  9.],
           [ 1.,  4.,  5., 16., 20., 25.]])
    >>> poly = PolynomialFeatures(interaction_only=True)
    >>> poly.fit_transform(X)
    array([[ 1.,  0.,  1.,  0.],
           [ 1.,  2.,  3.,  6.],
           [ 1.,  4.,  5., 20.]])
    """

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
        n_out = n_int_cols*(n_samples+1)
        XP = np.empty(
            shape=(n_samples, n_out), dtype=X.dtype, order=self.order
        )

        XP[:, 0:len(self.interaction_columns)] = (
            X[:, [self.interaction_columns]])
        for index in range(X.shape[1]):
            for int_col in self.interaction_columns:
                XP[:, index+n_int_cols] = X[:, index] * X[:, int_col]

        return XP
