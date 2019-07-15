import os
import models.args


def get_args():
    parser = models.args.get_args()

    parser.add_argument('--dataset', type=str, default='Reuters', choices=['Reuters', 'AAPD', 'IMDB', 'Yelp2014', 'MBTI'])
    parser.add_argument('--penalty', type=str, default='l2', choices=['l2', 'l1'])
    parser.add_argument('--C', type=float, default=0.25)
    parser.add_argument('--tol', type=float, default=0.0001)
    parser.add_argument('--max_iter', type=int, default=100000)
    parser.add_argument('--embedding', type=str, default='binary', choices=['binary', 'tfidf', 'word2vec'])
    parser.add_argument('--word-vectors-file', default='GoogleNews-vectors-negative300.txt')
    parser.add_argument('--save-path', type=str, default=os.path.join('model_checkpoints', 'nbsvm'))
    parser.add_argument('--trained-model', type=str)

    args = parser.parse_args()
    return args
