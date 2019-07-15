import numpy as np
from scipy.sparse import spmatrix, coo_matrix
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model.base import SparseCoefMixin
from sklearn.utils.validation import check_X_y, check_is_fitted


class NBSVM(BaseEstimator, ClassifierMixin,SparseCoefMixin):
    """
    NBSVM is a strong linear meta-classifer proposed by Manning at ACL 2012.
    This implementation is based on code shared by T. Howard

    You must embed input data, the schema is up to the user, but we recommend
    either binary ot rfidf:

    Vectorizer(
        analyzer="word",
        ngram_range=(1,2),
        binary=True,
        min_df=3,
        max_df=0.9,
        max_features=25000,
        lowercase=False,
        stop_words=SPACY_STOPWORDS,
        )

    Or, following the original paper:

     TfidfVectorizer(
        ngram_range=(1,2),
        min_df=3,
        max_df=0.9,
        strip_accents='unicode',
        use_idf=1,
        smooth_idf=1,
        sublinear_tf=1,
        )

    You may also use word2vec, glove, or anything else, of course.
    """

    def __init__(self, config,):
        """
        Constructs an instance of NBSVM multilabel classifier.
        """
        self.C = config.C
        self.dual = config.dual
        self.n_jobs = config.n_jobs
        self.solver = config.solver
        self.penalty = config.penalty
        self.max_iter = config.max_iter
        self.tol = config.tol
        self.vectors_path = config.vectors_path
        if config["pretrained_model"] is not None:
            self.model  = joblib.load("pretrained_model")
        else:
            self.model  =  None

    def predict(self, x):
        """
        Predict labels based on input data
        """
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict(x.multiply(self._r))

    def predict_proba(self, x):
        """
        Predict probabilities based on input data
        """
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict_proba(x.multiply(self._r))

    def fit(self, x, y):
        """
        Fits data to a model, return self.
        params:
            x: (n,m)-shaped matrix of independent variables
            y: (n,m)-shaped matrix of dependant variables
        """
        # Check that X and y have correct shape
        x, y = check_X_y(x, y, accept_sparse=True)
        self.classes_ = np.unique(y)
        def pr(x, y_i, y):
            p = x[y==y_i].sum(0)
            return (p+1) / ((y==y_i).sum()+1)

        self._r = sparse.csr_matrix(np.log(pr(x,1,y) / pr(x,0,y)))
        x_nb = x.multiply(self._r)
        self._clf = LogisticRegression(C=self.C, dual=self.dual, solver=self.solver, tol=self.tol, max_iter=self.max_iter, n_jobs=self.n_jobs, penalty=self.penalty).fit(x_nb, y)
        return self
