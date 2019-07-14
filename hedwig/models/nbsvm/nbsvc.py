import numpy as np
from scipy.sparse import spmatrix, coo_matrix
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.linear_model.base import LinearClassifierMixin, SparseCoefMixin
from sklearn.svm import LinearSVC
from sklearn.utils.validation import check_is_fitted


class NBSVC(BaseEstimator, LinearClassifierMixin, SparseCoefMixin):

    def __init__(self,
                 loss='squared_hinge',
                 penalty='l2',
                 laplace=0.1,
                 beta=0.25,
                 C=1,
                 tol=1e-4,
                 max_iter=100000,
                ):
        """
        Constructs a NBLinearSVClassifier classifer that implements a NB-SVM.

        This implementation is an improvement on what can be described as iterative improvements over the original paper.
        On the same data, we consistently obtain roughly 2-2.5% higher results, reducing the error rate by nearly 20%.

        Differences from the publication include but note limited to:
          1) Binary Count Vectorization schema is used by default instead of TfIdf. Consistently produces bettter results across all datasets tested.
          2) LIBLINEAR instead of LibSVM (no kernel trick). Faster & more accurate for text classification (and it alone, LibSVM is better overall)
          3) Intercept is directly calculated, no hyperplane fitting via logit (as done by J. Howard for Toxic Comment @ Kaggle)

        Parameters:
          loss: hinge (classic svm) or squared_hinge (default, results in smoother solution)
          penalty: l2 or l1, default is l2 (l1 tends to produce more sparse solutions, but with binary vectorizer we are already sparse)
          laplace: laplace smoothing, (alpha in original paper) (0-1]: used in MNB, if < 1 mitigates imbalance in the distribution. Set to 1 otherwise.
          beta: Interpolation factor, [0.0-1.0],  that determines significance given SVM vs NB.  1 means all SVM, 0 all NB
          C: SVM's soft margin tolerance parameter.  As we decrease C we increase model complexity and vice versa.
          tol: tolerance to unsound solution when determining stopping. If you keep getting differnt results on same data, lower this parameter (will tke longer)
          max_iter: how many iterations LIBLINEAR's optimizer is allowed to go through before giving up (more results in better solutions and can overfit, takes longer)
        """
        self.loss_ = loss_
        self.penalty_ = penalty
        self.laplace_ = laplace
        self.beta_ = beta
        self.C_ = C
        self.tol_ = tol
        self.max_iter_ = max_iter
        self.classes_ = None


    def fit(self, X, y):
        """
        Train the model. Takes an (m,n) shaped matrix, and (m,l) where l is number of labels. Returns self. You can then use predict(X)
        Parameters:
          X: independent features in matrix-like format, such as numpy or dataframe
          y: dependent feature(s) in same format as X
        """

        self.classes_ = np.unique(y)
        if len(self.classes_) == 2:
            coef_, intercept_ = self._fit_binary(X, y)
            self.coef_ = coef_
            self.intercept_ = intercept_
        else:
            coef_, intercept_ = zip(*[
                self._fit_binary(X, y == class_)
                for class_ in self.classes_
            ])
            self.coef_ = np.concatenate(coef_)
            self.intercept_ = np.array(intercept_).flatten()
        return self

    def _fit_binary(self, X, y):
        p = np.asarray(self.laplace_ + X[y == 1].sum(axis=0)).flatten()
        q = np.asarray(self.laplace_ + X[y == 0].sum(axis=0)).flatten()
        r = np.log(p/np.abs(p).sum()) - np.log(q/np.abs(q).sum())
        b = np.log((y == 1).sum()) - np.log((y == 0).sum())

        if isinstance(X, spmatrix):
            indices = np.arange(len(r))
            r_sparse = coo_matrix(
                (r, (indices, indices)),
                shape=(len(r), len(r))
            )
            X_scaled = X * r_sparse
        else:
            X_scaled = X * r

            lsvc = LinearSVC(
                 C=self.C_,
                 penalty=self.penalty_,
                 loss=self.loss_,
                 class_weight=None,
                 tol=self.tol_,
                 fit_intercept=True,
                 intercept_scaling=1.0,
                 max_iter=self.max_iter_,
        ).fit(X_scaled, y)

        mean_mag =  np.abs(lsvc.coef_).mean()

        coef_ = (1 - self.beta_) * mean_mag * r + \
                self.beta_ * (r * lsvc.coef_)

        intercept_ = (1 - self.beta_) * mean_mag * b + \
                     self.beta_ * lsvc.intercept_

        return coef_, intercept_
