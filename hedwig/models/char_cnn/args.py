import os

import models.args


def get_args():
    parser = models.args.get_args()

    parser.add_argument('--dataset', type=str, default='Reuters', choices=['Reuters', 'AAPD', 'IMDB', 'Yelp2014', 'MBTI'])
    parser.add_argument('--num-conv-filters', type=int, default=256)
    parser.add_argument('--num-affine-neurons', type=int, default=1024)
    parser.add_argument('--output-channel', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epoch-decay', type=int, default=3)
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--save-path', type=str, default=os.path.join('model_checkpoints', 'char_cnn'))
    parser.add_argument('--resume-snapshot', type=str)
    parser.add_argument('--trained-model', type=str)

    args = parser.parse_args()
    return args
