## HedwigANLP

This repo is a fork of Hedwig, a library containing PyTorch deep learning models for document classification.
The original was implemented by the Data Systems Group at the University of Waterloo [git source](https://github.com/castorini/hedwig.git)
This fork adds a few models and datasets. It was developed as part of research with Arjun Mukherjee's group.
Corresponding authors: Dainis Boumber, dainis.boumber@gmail.com

#### Additions in this fork:

+ MBTI Dataset and all the necessary modules
+ utility to preprocess and add new datasets saved from regular Pandas dataframe
+ a few bug fixes

#### Coming Soon

+ Support for PyTorch 1.0/1.1
+ Support for Python 3.7
+ Support for mixed precision training for all models
+ Distributed training for models that need it
+ Dedicated embeddings module
+ More automation to dataset addition process
+ Several SOTA and baselines classifiers

#### Models

+  HBERT: Hierarchical BERT for finetuning on Document Classification tasks.
+ [DocBERT](models/bert/) : DocBERT: BERT for Document Classification [(Adhikari et al., 2019)](https://arxiv.org/abs/1904.08398v1)
+ [Reg-LSTM](models/reg_lstm/): Regularized LSTM for document classification [(Adhikari et al., NAACL 2019)](https://cs.uwaterloo.ca/~jimmylin/publications/Adhikari_etal_NAACL2019.pdf)
+ [XML-CNN](models/xml_cnn/): CNNs for extreme multi-label text classification [(Liu et al., SIGIR 2017)](http://nyc.lti.cs.cmu.edu/yiming/Publications/jliu-sigir17.pdf)
+ [HAN](models/han/): Hierarchical Attention Networks [(Zichao et al., NAACL 2016)](https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf)
+ [Char-CNN](models/char_cnn/): Character-level Convolutional Network [(Zhang et al., NIPS 2015)](http://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf)
+ [Kim CNN](models/kim_cnn/): CNNs for sentence classification [(Kim, EMNLP 2014)](http://www.aclweb.org/anthology/D14-1181)

Each model directory has a `README.md` with further details.

#### Setting up PyTorch

Hedwig was designed for Python 3.6 and [PyTorch](https://pytorch.org/) 0.4. 

PyTorch recommends [Anaconda](https://www.anaconda.com/distribution/) for managing your environment.
We'd recommend creating a custom environment as follows:

```
$ conda create --name castor python=3.6
$ source activate castor
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

Download the Reuters, AAPD and IMDB datasets, along with word2vec embeddings from
[`hedwig-data`](https://git.uwaterloo.ca/jimmylin/hedwig-data).

```bash
$ git clone https://github.com/castorini/hedwig.git
$ git clone https://git.uwaterloo.ca/jimmylin/hedwig-data.git
```

Organize your directory structure as follows:

```
.
├── hedwig
└── hedwig-data
```

After cloning the hedwig-data repo, you need to unzip the embeddings and run the preprocessing script:

```bash
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

Likewise, you can add different embeddings in the same manner to `hedwig-data/embeddings`. Just don't forget to tell the model what to use in command line args

