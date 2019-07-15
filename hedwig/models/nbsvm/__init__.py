"""
NB-SVM
Author: Dainis Boumber, <dainis.boumber@gmail.com>

Based on 'Baselines and Bigrams: Simple, Good Sentiment and Topic Classification',  Wang, Sida I. and Manning, Christopher D, ACL'2012
For details see: https://github.com/sidaw/nbsvm/blob/master/wang12simple.pdf
We would also likr to credit J. Howard for sharing initial probability estimation code with the public.

Overview:
  A NB-SVM generates additional Bayesian meta-features, and classifies by balancing between MNB and Linear SVM.
  It tends to have the best of both of these models. In (S. Wang and C. Manning, 2012) it was shown to significntly
  outperform both MNB and Linear SVM when tested on a collection of varying datasets; it also dominated all otherwise
  strong linear baselines.

NB-SVM performance:

  For single-label tasks:NB-SVM is nearly certainly the strongest linear model across the board and can be tough to beat.
  It remains competitive against common deep models when carefuly engineered or ensembled, but is best used as a strong baseline.

  For multi-label tasks:these models are still very close to state-of-the art.
  NBSVMs do not care about class imbalance much. This package can an handle sparse matrices.

Usage:
  implementation that fully conforms to the scikit-learn protocol. E.g. you can use fit-predict routines, or you can use otherwise
  with almost anything within scikit-learn and expect it to behave just like any other built-in sklearn classifer would.
"""

__all__ = ['NBSVM']
