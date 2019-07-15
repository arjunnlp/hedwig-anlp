## HedwigANLP

This repo is based on a fork of Hedwig and Hedwig-Data, a library containing PyTorch deep learning models for document classification and various datasets
It also contains a number of other classifiers developed and datasets or implemented by the corresponding authors and other lab members, as well as
modifications, extensions and bug fixes of existing code. The algorithms within are most suitable for binary/multiclass/multilabel, longer text,
single-sequence classification tasks.

The original Hedwig was implemented by the Data Systems Group at the University of Waterloo [git source](https://github.com/castorini/hedwig.git)
HedwigANLP is being developed as part of research with Arjun Mukherjee's group.

Corresponding authors (feature requests, bug reports): Dainis Boumber, dainis.boumber@gmail.com

*Note: additional documentation is present throughout the library, in both `hedwig` and and `hedwig-data` directories, in the form of README.md files.*

#### Setup

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

```
$ python
>>> import nltk
>>> nltk.download()
```

Furthermore, you want to get spacy stopwords and models

```
$ spacy download en
```

Spacy stop words tend to be superior to those offered by NLTK, sklearn, or StanfordNLP,
although it's task-specific. They are accessible like so:

```
spacy_nlp = spacy.load('en_core_web_sm')
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
```

#### Models

+ [HBERT](hedwig/models/hbert/): Hierarchical BERT for finetuning on Document Classification tasks.
+ [DocBERT](hedwig/models/bert/): DocBERT: BERT for Document Classification [(Adhikari et al., 2019)](https://arxiv.org/abs/1904.08398v1)
+ [Reg-LSTM](hedwig/models/reg_lstm/): Regularized LSTM for document classification [(Adhikari et al., NAACL 2019)](https://cs.uwaterloo.ca/~jimmylin/publications/Adhikari_etal_NAACL2019.pdf)
+ [XML-CNN](hedwig/models/xml_cnn/): CNNs for extreme multi-label text classification [(Liu et al., SIGIR 2017)](http://nyc.lti.cs.cmu.edu/yiming/Publications/jliu-sigir17.pdf)
+ [HAN](hedwig/models/han/): Hierarchical Attention Networks [(Zichao et al., NAACL 2016)](https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf)
+ [Char-CNN](hedwig/models/char_cnn/): Character-level Convolutional Network [(Zhang et al., NIPS 2015)](http://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf)
+ [Kim CNN](hedwig/models/kim_cnn/): CNNs for sentence classification [(Kim, EMNLP 2014)](http://www.aclweb.org/anthology/D14-1181)
+ [NBSVM](hedwig/models/nbsvm/): Two complimentary implementations of NBSVM, shown to be the strongest linear baseline all-around. [(S. Wang and C. Manning)](https://github.com/sidaw/nbsvm/blob/master/wang12simple.pdf)

Each model directory has a `README.md` with further details. All follow similar training pattern (differences are explained in their specific `README.md` files.
Training is simple. For example, if you are using XML-CNN on MBTI dataset you would do something similar to this (these are of course not optimal hypermarameters):

```
python -m models.xml_cnn --mode non-static --dataset MBTI --batch-size 1024 --lr 0.01 --epochs 30 --dropout 0.5 --dynamic-pool-length 8 --seed 3435
```

These are of course sub-obtimal hyperparameters, just an example. Better results can be achieved by tuning various knobs, for example:

```
python -m models.xml_cnn --dataset MBTI --mode rand --batch-size 512 --lr 0.002 --epoch-decay 2 --dev-every 2 --epochs 10 --dropout 0.33 --dynamic-pool-length 16 --seed 3435
```

`--mode` hyper-parameter:

***rand** makes hedwig train the embeddings, **non-static** makes hedwig fine-tune existing pre-trained embeddings (typically ones you specify with this task in mind), **static** runs on pre-trained without modifying them, **multichannel or not specifying mode option** for the CNN models leads to multichannel training with one channel being static and the other one being fine-tuneable*

results in micro-F1 of 0.76 vs 0.72 for the first example, and can be done with a smaller GPU due to twice smaller batch size. In general, bigger batch size leads to smoother optimization, but will not always give best results if the other parameters are not scaled correctly. For further discussion on this topic, we refer you to the following publications:

[A Disciplined Approach to Neural Network Hyper-Parameters](https://arxiv.org/pdf/1803.09820.pdf) by [Leslie Smith](https://arxiv.org/search/cs?searchtype=author&query=Smith%2C+L+N)
[Don't Decay the Learning Rate, Increase the Batch Size](https://openreview.net/pdf?id=B1Yy1BxCZ) by [Stephen L. Smith et al.](https://arxiv.org/search/cs?searchtype=author&query=Smith%2C+S+L)

For more information regarding each architecture, look inside each model's directory for a file called `args.py` that describes to see what parameters it takes, in addition to standard ones like learning rate and batch size which are shown in `models/args.py`
Each model may have additional command line arguments you can use -- in this example I am only showing a few, for XML-CNN, for example, there is around two dozen things you can tune.

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

Will get you the MBTI data which has been "twitterized", e.g. preprocessed in the same manner Stanford NLP team did glove-twitter-200 data.

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

Summary: your dataset must comnform to spcifications defined by `torchtext.data` and `torchtext.datasets` - see [torchtext documentation](https://torchtext.readthedocs.io/en/latest/) for a detailed guide.

+ Add a directory named after dataset name in `hedwig-data/datasets/` Within it, you want to have 3 files: train.tsv, test.tsv, and dev.tsv.
+ Use `add_dataset.ipynb` notebook, found in the `hedwig/utils/` directory, to pre-process your Pandas dataframe into tsv file that can be used by Hedwig.
+ Preprocessing is of course task-dependent, for examples see other datasets in `hedwig-data`, `datasets/` directory, and `utils` directory, namely `utils/add_dataset.ipynb`, `utils/add_dataset.py`, `utils/twitterize.py`.
+ Add the code necessary to load, process, train and evaluate on your new dataset throughout the library. Small modifications may be necessary to roughly 25% of the library, but they are really simple to do. See how MBTI was added for an example, and copy-paste (while changing relevant things like number of labels, classes, etc.). You will want to add things, specifically, to `datsets`, each of the model's `__main__.py` and `args.py` files, and a few other places.

**You want to have the data in TSV format. Make sure that your data does not contain " anywhere, as well as escape characters or invalid unicode, that the label column is separated from text by a tab, that neither label nor text is surrounded by quotation marks, and that there is only one \n -- at the end of each line**

See `utils/add_dataset.ipynb` for how to do it if you encounter issues.
Your preprocessing should, ideally, take into account the word embeddings used (by default most things not BERT are taking in word2vec, which has very specific preprocessing rules).

If you want to add, change or remove metrics, see `hedwig/common/` directory.

#### Using different embeddings

Likewise, you can add different embeddings in the same manner to `hedwig-data/embeddings`. Just don't forget to tell the model what to use in command line args. Example using glove-twitter-200:

```
python -m models.xml_cnn --mode non-static  --dataset MBTI --batch-size 1024 --lr 0.002 --epochs 10 --dropout 0.7 --seed 3435 --word-vectors-dir ../hedwig-data/embeddings/glove-twitter-200 --word-vectors-file glove-twitter-200.txt --embed-dim 200 --words-dim 200 --weight-decay 0.0001

```

#### Recent additions:

+ MBTI Dataset and all the necessary modules
+ Utilities to preprocess and add new datasets saved from regular Pandas dataframe
+ Tokenizer and preprocessor that follows protocols used by StanfordNLP when making glove-twitter-200
+ Many bug fixes

#### TODO

+ Integrate preprocessing from totchtext and phase out use of alternatives when possible
+ Support for PyTorch 1.0/1.1
+ Support for Python 3.7
+ Support for mixed precision training for all models
+ NBSVM

#### Coming Soon

+ Distributed training for models that need it
+ Dedicated embeddings module
+ More automation to dataset addition process
+ Several SOTA classifiers
