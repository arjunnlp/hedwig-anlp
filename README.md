## HedwigANLP

This repo is based on a fork of Hedwig and Hedwig-Data, a library containing PyTorch deep learning models for document classification and various datasets
It also contains a number of other classifiers developed and datasets or implemented by the corresponding authors and other lab members, as well as
modifications, extensions and bug fixes of existing code. The algorithms within are most suitable for binary/multiclass/multilabel, longer text,
single-sequence classification tasks.

The original Hedwig was implemented by the Data Systems Group at the University of Waterloo [git source](https://github.com/castorini/hedwig.git)
HedwigANLP is being developed as part of research with Arjun Mukherjee's group.

Corresponding authors (feature requests, bug reports): Dainis Boumber, dainis.boumber@gmail.com

*Note: additional documentation is present throughout the library, in both `hedwig` and and `hedwig-data` directories, in the form of README.md files.*

#### Models

+  HBERT: Hierarchical BERT for finetuning on Document Classification tasks.
+ [DocBERT](models/bert/) : DocBERT: BERT for Document Classification [(Adhikari et al., 2019)](https://arxiv.org/abs/1904.08398v1)
+ [Reg-LSTM](models/reg_lstm/): Regularized LSTM for document classification [(Adhikari et al., NAACL 2019)](https://cs.uwaterloo.ca/~jimmylin/publications/Adhikari_etal_NAACL2019.pdf)
+ [XML-CNN](models/xml_cnn/): CNNs for extreme multi-label text classification [(Liu et al., SIGIR 2017)](http://nyc.lti.cs.cmu.edu/yiming/Publications/jliu-sigir17.pdf)
+ [HAN](models/han/): Hierarchical Attention Networks [(Zichao et al., NAACL 2016)](https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf)
+ [Char-CNN](models/char_cnn/): Character-level Convolutional Network [(Zhang et al., NIPS 2015)](http://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf)
+ [Kim CNN](models/kim_cnn/): CNNs for sentence classification [(Kim, EMNLP 2014)](http://www.aclweb.org/anthology/D14-1181)

Each model directory has a `README.md` with further details. All follow similar training pattern (differences are explained in their specific `README.md` files.
Training is simple. For example, if you are using XML-CNN on MBTI dataset you would do something similar to this (these are of course not optimal hypermarameters):

```
python -m models.xml_cnn --mode non-static --dataset MBTI --batch-size 1024 --lr 0.01 --epochs 30 --dropout 0.5 --dynamic-pool-length 8 --seed 3435
```

These are of course sub-obtimal hyperparameters, just an example. Each model has a file called `args.py` that you can look into to see what parameters it takes, in addition to standard ones like learning rate and batch size.
Each model may have additional command line arguments you can use -- in this example I am only showing a few, for XML-CNN, for example, there is around two dozen things you can tune.

#### Setting up PyTorch

Hedwig was designed for Python 3.6 and [PyTorch](https://pytorch.org/) 0.4.1

PyTorch recommends [Anaconda](https://www.anaconda.com/distribution/) for managing your environment.
We'd recommend creating a custom environment as follows:

```
$ conda create --name hedwig python=3.6
$ conda activate hedwig
```

And installing PyTorch as follows:

```
$ conda install pytorch=0.4.1 cuda92 -c pytorch
```

Other Python packages we use can be installed via pip:

```
$ pip install -r requirements.txt
```

Code depends on data from NLTK (e.g., stopwords) so you'll have to download them.
Run the Python interpreter and type the commands:

```python
>>> import nltk
>>> nltk.download()
```

#### Datasets

You want to organize your directories as follows:

Organize your directory structure as follows:

```
hedwig-anlp
           ├── hedwig
           └── hedwig-data
```

This is done for you in this repository already, but double-check as a sanity measure.

`hedwig-data`, complete with default embeddings and glove-twitter-200 embeddings, default datasets and additional one called MBTI (for Marjan's paper, setup in Hedwig format and ready for use) can be found on backblaze account I setup a while ago -- search e-mail for logon credentials or ask me. Ideally you want to store stuff there, since it takes seconds to upload/dowbload 20-30GB csv, whereas Google Drive sometimes has issues with that. Plus, it's free 10 TB storage.

I had already setup access for big-box-1 and will follow up with the server and the other box. Use it like so:

```
$ b2
```

That will produce a list of commands and explanations. Most of hedwig-related stuff is in the bucket called "marjan"

```
$ b2 download-file-by-name marjan hedwig-data.tar.gz
```

Will get the hedwig-data directory.

```
$ b2 download-file-by-name marjan twitter.tar.gz
```

Will get you the MBTI data which has been "twitterized", e.g. preprocessed in the same manner Stanford NLP team did glove-twitter-* data.

Alternative to the above approach is to download the Reuters, AAPD and IMDB datasets, along with word2vec embeddings from hedwig-data fomr github of University of Waterloo:

```
$ git clone https://git.uwaterloo.ca/jimmylin/hedwig-data.git
```

After cloning the hedwig-data repo, you need to unzip the embeddings and run the preprocessing script:

```
cd hedwig-data/embeddings/word2vec
gzip -d GoogleNews-vectors-negative300.bin.gz
python bin2txt.py GoogleNews-vectors-negative300.bin GoogleNews-vectors-negative300.txt
```

#### Adding new datasets

Add a directory named after dataset name in hedwig-data/datasets/
Within it, you will have 3 files: train.tsv, test.tsv, and dev.tsv.
Use add_dataset.ipynb notebook, found in the utils/ directory, to pre-process your Pandas dataframe into tsv file that can be used by Hedwig.
Add the code necessary to load, process, train and evaluate on your new dataset. Small modifications may be necessary to roughly 25% of the library, but they are really simple to do. See how MBTI was added for an example, and copy-paste (while changing relevant things like number of labels, classes, etc.)
Your preprocessing should take into account the word embeddings used (by default most things not BERT are taking in word2vec, which has very specific preprocessing rules).

#### Using different embeddings

Likewise, you can add different embeddings in the same manner to `hedwig-data/embeddings`. Just don't forget to tell the model what to use in command line args. Example using glove-twitter-200:

```
python -m models.xml_cnn --mode non-static  --dataset MBTI --batch-size 1024 --lr 0.002 --epochs 10 --dropout 0.7 --seed 3435 --word-vectors-dir ../hedwig-data/embeddings/glove-twitter-200 --word-vectors-file glove-twitter-200.txt --embed-dim 200 --words-dim 200 --weight-decay 0.0001

```

#### Recent additions:

+ MBTI Dataset and all the necessary modules
+ utility to preprocess and add new datasets saved from regular Pandas dataframe
+ a few bug fixes

#### TODO

+ Support for PyTorch 1.0/1.1
+ Support for Python 3.7
+ Support for mixed precision training for all models
+ NBSVM

#### Coming Soon

+ Distributed training for models that need it
+ Dedicated embeddings module
+ More automation to dataset addition process
+ Several SOTA classifiers
