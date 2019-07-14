import numpy as np
from scipy.sparse import spmatrix, coo_matrix
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model.base import SparseCoefMixin
from sklearn.utils.validation import check_X_y, check_is_fitted


class NBSVM(BaseEstimator, ClassifierMixin,SparseCoefMixin):
    """
    An evolution of NB-SVM implementation by Jeremy Howard. This version is slightly worse than NBLinearSVClassifier in binary and multiclass problems,
    but may have an edge in mult-label tasks. It also natively computes probabilites and allows for AUC. NBLinearSVClassifier is incapable of either
    and can only deliver those via a stand-alone cross-validating calibrator, a costly and involved process with dubious outcomes when it comes to
    """
    def __init__(self, laplace=1, C=1.0, tol=0.0001, max_iter=1000000):
        """
        Constructs a NB-SVM Classifier following Jeremy Howard's approach

        This NB-SVM is most suitable for multi-label and as it was constructedf with that problem in mind. Unlike NBSVC, here logit is used.
        Estimator is still LIBLINEAR, and we get probabilities (hence can do AUC and other important things). It also tends to work well when ensembling
        NB-SVC and NB-SVM and using them in  error corrector ensembling approach (just voting meta-classification)

        Parameters:
          laplace: laplace smoothing, (alpha in original paper) (0-1]: used in MNB, if < 1 mitigates imbalance in the distribution. Set to 1 otherwise.
          C: SVM's soft margin tolerance parameter.  As we decrease C we increase model complexity and vice versa.
          tol: tolerance to unsound solution when determining stopping. If you keep getting differnt results on same data, lower this parameter (will tke longer)
          max_iter: how many iterations LIBLINEAR's optimizer is allowed to go through before giving up (more results in better solutions and can overfit, takes longer)
        """
        self.laplace_ = laplace
        self.C_ = C
        self.tol_ = tol
        self.max_iter_ = max_iter
        self.risk_ = None
        self.model_ = None


    def predict(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['risk_', 'model_'])
        return self.model_.predict(x.multiply(self.risk_))

    def predict_proba(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['risk_', 'model_'])
        return self.model_.predict_proba(x.multiply(self.risk_))

    def fit(self, x, y):
        # Check that X and y have correct shape
        x, y = check_X_y(x, y, accept_sparse=True)
        self.classes_ = np.unique(y)
        def pr(x, y_i, y):
            p = x[y==y_i].sum(0)
            return (p+self.laplace_) / ((y==y_i).sum()+self.laplace_)

        self.risk_ = sparse.csr_matrix(np.log(pr(x,1,y) / pr(x,0,y)))
        x_nb = x.multiply(self.risk_)
        self.model_ = LogisticRegression(C=self.C_, dual=True, solver='liblinear', tol=self.tol_, max_iter=self.max_iter_, n_jobs=1, penalty='l2').fit(x_nb, y)
        return self
