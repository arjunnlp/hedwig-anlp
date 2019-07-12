# Hedwig-ANLP
Deep text/document classification for binary/multiclass/multilabel, single-sequence tasks. Forked from Hedwig. 
Hedwig is a collection of PyTorch deep learning models implemented by the Data Systems Group at the University of Waterloo.

#### Models
- HBERT: Hierarchical BERT for finetuning on Document Classification tasks.
- DocBERT : BERT for Document Classification (Adhikari et al., 2019)
- Reg-LSTM: Regularized LSTM for document classification (Adhikari et al., NAACL 2019)
- XML-CNN: CNNs for extreme multi-label text classification (Liu et al., SIGIR 2017)
- HAN: Hierarchical Attention Networks (Zichao et al., NAACL 2016)
- Char-CNN: Character-level Convolutional Network (Zhang et al., NIPS 2015)
- Kim CNN: CNNs for sentence classification (Kim, EMNLP 2014)

Each model directory has a README.md with further details.

#### Setting up PyTorch
Hedwig is designed for Python 3.6 and PyTorch 0.4. PyTorch recommends Anaconda for managing your environment. We'd recommend creating a custom environment as follows:

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

Code depends on data from NLTK (e.g., stopwords) so you'll have to download them. Run the Python interpreter and type the commands:

```
>>> import nltk
>>> nltk.download()
```

#### Datasets
Download the Reuters, AAPD and IMDB datasets, along with word2vec embeddings from hedwig-data.

```
$ git clone https://git.uwaterloo.ca/jimmylin/hedwig-data.git
```

Organize your directory structure as follows:

hedwig-anlp
           ├── hedwig
           └── hedwig-data
           
After cloning the hedwig-data repo, you need to unzip the embeddings and run the preprocessing script:

```
cd hedwig-data/embeddings/word2vec 
gzip -d GoogleNews-vectors-negative300.bin.gz 
python bin2txt.py GoogleNews-vectors-negative300.bin GoogleNews-vectors-negative300.txt 
```

Adding new datasets:

1) Add a directory named after dataset name in `hedwig-data/datasets/`
2) Within it, you will have 3 files: train.tsv, test.tsv, and dev.tsv.
3) Use `add_dataset.ipynb` notebook, found in the `utils/` directory, to pre-process your Pandas dataframe into tsv file that can be used by Hedwig.
4) Add the code necessary to load, process, train and evaluate on your new dataset. Small modifications may be necessary to roughly 25% of the library, but they are really simple to do. See how MBTI was added for an example, and copy-paste (while changing relevant things like number of labels, classes, etc.)

Your preprocessing should take into account the word embeddings used (by default most things not BERT are taking in word2vec, which has very specific preprocessing rules).


