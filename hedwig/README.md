## Hedwig

This repo is a fork of Hedwig, a library containing PyTorch deep learning models for document classification.
The original was implemented by the Data Systems Group at the University of Waterloo [git source](https://github.com/castorini/hedwig.git)
This fork adds a few models and datasets. It was developed as part of research with Arjun Mukherjee's group.
Corresponding authors: Dainis Boumber, dainis.boumber@gmail.com

*More comprehensive documentation is located in the parent directory*

#### Models

+  HBERT: Hierarchical BERT for finetuning on Document Classification tasks.
+ [DocBERT](models/bert/) : DocBERT: BERT for Document Classification [(Adhikari et al., 2019)](https://arxiv.org/abs/1904.08398v1)
+ [Reg-LSTM](models/reg_lstm/): Regularized LSTM for document classification [(Adhikari et al., NAACL 2019)](https://cs.uwaterloo.ca/~jimmylin/publications/Adhikari_etal_NAACL2019.pdf)
+ [XML-CNN](models/xml_cnn/): CNNs for extreme multi-label text classification [(Liu et al., SIGIR 2017)](http://nyc.lti.cs.cmu.edu/yiming/Publications/jliu-sigir17.pdf)
+ [HAN](models/han/): Hierarchical Attention Networks [(Zichao et al., NAACL 2016)](https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf)
+ [Char-CNN](models/char_cnn/): Character-level Convolutional Network [(Zhang et al., NIPS 2015)](http://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf)
+ [Kim CNN](models/kim_cnn/): CNNs for sentence classification [(Kim, EMNLP 2014)](http://www.aclweb.org/anthology/D14-1181)
+ NBSVM: Naive Bayes SVM for multi-class multi-label hierarchical text classification, follows description by Hinton in one of his lectures. One of the strongest models in practice.

Training is simple. For example, if you are using XML-CNN on MBTI dataset you would do something similar to this (these are of course not optimal hypermarameters):

```
python -m models.xml_cnn --mode non-static --dataset MBTI --batch-size 1024 --lr 0.01 --epochs 30 --dropout 0.5 --dynamic-pool-length 8 --seed 3435
```

Each model may have additional command line arguments you can use -- in this example I am only showing a few, for XML-CNN, for example, there is around two dozen things you can tune. Each model directory has a `README.md` with further details.

#### Preprocessing MBTI and new datasets

This is of course task-dependent, for examples see other datasets in `hedwig-data`, `datasets/` directory, and `utils` directory, namely `utils/add_dataset.ipynb`, `utils/add_dataset.py`, `utils/twitterize.py`.
You want to have the data in TSV format.
**Make sure that your data does not contain " anywhere, as well as escape characters or invalid unicode, that the label column is separated from text by a tab, that neither label nor text is surrounded by quotation marks, and that there is only one \n -- at the end of each line**
See `utils/add_dataset.ipynb` for how to do it if you encounter issues. 
