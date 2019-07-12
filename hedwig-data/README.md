To add a new dataset - for example called A - make a directory like so:

```
mkdir datasets/A
```

Then place your data as**train.tsv, dev.tsv, and test.tsv** inside the directory, having first made sure to preprocess it using a notebook for adding datasets that is located in `hedwig/utils`.

You will also need to add some code to Hedwig library so that it will recognize your dataset and load it properly. See how MBTI was done.
