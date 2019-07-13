import pandas as pd
import numpy as np
from pathlib import Path
import ftfy
import textacy
import csv
from gensim.utils import simple_preprocess
import unicodedata
import swifter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import sys
import regex as re
from argparse import ArgumentParser


PATH_TO_DATASETS=Path('../../hedwig-data/datasets')
stop_words = list(set(stopwords.words('english')))
FLAGS = re.MULTILINE | re.DOTALL

def hashtag(text):
    text = text.group()
    hashtag_body = text[1:]
    if hashtag_body.isupper():
        result = " {} ".format(hashtag_body.lower())
    else:
        result = " ".join([""] + [re.sub(r"([A-Z])",r" \1", hashtag_body, flags=FLAGS)])
    return result

def allcaps(text):
    text = text.group()
    return text.lower() + " "

def tweet_preprocessing(text):
    # Different regex parts for smiley faces
    eyes = r"[8:=;]"
    nose = r"['`-]?"

    # function so code less repetitive
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)

    text = text.replace("@ HASHTAG ", "#")
    text = text.replace("@ USER ", "<user> ")
    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
    text = re_sub(r"@\w+", "<user>")
    text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<smile>")
    text = re_sub(r"{}{}p+".format(eyes, nose), "<lolface>")
    text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<sadface>")
    text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutralface>")
    text = re_sub(r"/"," / ")
    text = re_sub(r"<3","<heart>")
    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
    text = re_sub(r"#\S+", hashtag)
    text = re_sub(r"([!?.]){2,}", r"\1 <repeat>")
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")
    text = text.replace("URL", "<url>")
    text = re_sub(r"([A-Z]){2,}", allcaps)
    
    return text.lower()
                 
def is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False

def is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False

def clean_text(text):
    """Performs invalid character removal and whitespace cleanup on text."""
    output = []
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xfffd or is_control(char):
            continue
        if is_whitespace(char):
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)

def fix_contractions(text):
    text = re.sub(
        r"(\b)([Aa]re|[Cc]ould|[Dd]id|[Dd]oes|[Dd]o|[Hh]ad|[Hh]as|[Hh]ave|[Ii]s|[Mm]ight|[Mm]ust|[Ss]hould|[Ww]ere|[Ww]ould) n't",
        r"\1\2 not",
        text,
    )
    text = re.sub(
        r"(\b)([Hh]e|[Ii]|[Ss]he|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou) 'll",
        r"\1\2 will",
        text,
    )
    text = re.sub(r"(\b)([Tt]here|[Hh]ere) 's", r"\1\2 is", text)
    text = re.sub(r"(\b)([Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou) 're", r"\1\2 are", text)
    text = re.sub(
        r"(\b)([Ii]|[Ss]hould|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Ww]ould|[Yy]ou) 've",
        r"\1\2 have",
        text,
    )
    text = re.sub(
        r"(\b)([Hh]e|[Ii]|[Ss]he|[Tt]hey|[Ww]e|[Yy]ou) 'd",
        r"\1\2 would",
        text,
    )
    # non-standard
    text = re.sub(r"(\b)([Cc]a) n't", r"\1\2n not", text)
    text = re.sub(r"(\b)([Ii]) 'm", r"\1\2 am", text)
    text = re.sub(r"(\b)([Ll]et) 's", r"\1\2 us", text)
    text = re.sub(r"(\b)([Ww]) on't", r"\1\2ill not", text)
    text = re.sub(r"(\b)([Ss]) han't", r"\1\2hall not", text)
    text = re.sub(r"(\b)([Yy])(?: 'all|a 'll)", r"\1\2ou all", text)
    #####################################################
    text = re.sub(
        r"(\b)([Aa]re|[Cc]ould|[Dd]id|[Dd]oes|[Dd]o|[Hh]ad|[Hh]as|[Hh]ave|[Ii]s|[Mm]ight|[Mm]ust|[Ss]hould|[Ww]ere|[Ww]ould) n ' t",
        r"\1\2 not ",
        text,
    )
    text = re.sub(
        r"(\b)([Hh]e|[Ii]|[Ss]he|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou) ' ll ",
        r"\1\2 will ",
        text,
    )
    text = re.sub(r"(\b)([Tt]here|[Hh]ere) ' s ", r"\1\2 is", text)
    text = re.sub(r"(\b)([Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou) ' re ", r"\1\2 are", text)
    text = re.sub(
        r"(\b)([Ii]|[Ss]hould|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Ww]ould|[Yy]ou) ' ve ",
        r"\1\2 have ",
        text,
    )

    text = re.sub(
        r"(\b)([Hh]e|[Ii]|[Ss]he|[Tt]hey|[Ww]e|[Yy]ou) ' d ",
        r"\1\2 would ",
        text,
    )
    # non-standard
    text = re.sub(r"(\b)([Cc]a) n ' t ", r"\1\2n not ", text)
    text = re.sub(r"(\b)([Ii]) ' m ", r"\1\2 am ", text)
    text = re.sub(r"(\b)([Ll]et) ' s ", r"\1\2 us ", text)
    text = re.sub(r"(\b)([Ww]) on ' t ", r"\1\2ill not ", text)
    text = re.sub(r"(\b)([Ss]ha) n ' t ", r"\1\2ll not ", text)
    text = re.sub(r"(\b)([Yy])(?: ' all | a ' ll )", r"\1\2ou all ", text)
    text=text.replace(" 's ", "'s ").replace(" ' s ", "'s ").replace(" i ' m ", " i'm ")
    return text

def hard_preprocess(df):
    df.iloc[:,1]=df.iloc[:,1].swifter.apply(ftfy.fix_text)
    df.iloc[:,1]=df.iloc[:,1].swifter.apply(clean_text)
    df.iloc[:,1]=df.iloc[:,1].swifter.apply(lambda x: x.replace('"', '').replace("\n", " ").replace("\\",""))
    df.iloc[:,1]=df.iloc[:,1].swifter.apply(lambda text: textacy.preprocess_text(
        text, no_currency_symbols=True,no_urls=True,no_emails=True,no_phone_numbers=True,no_numbers=True))
    df.iloc[:,1] = df.iloc[:,1].apply(lambda x: x.replace("`","'").replace(
                                        "& amp ;", " and ").replace(
                                        "@ USER", "[USER]").replace(
                                        "@ URL", "[URL]").replace(
                                        "@ HASHTAG", "[HASHTAG]").replace(
                                        "*NUMBER*", "[NUMBER]")
                                     )
    df.iloc[:,1]=df.iloc[:,1].apply(fix_contractions)
    df.iloc[:,1] = df.iloc[:,1].swifter.apply(lambda text: " ".join(
        [word for word in word_tokenize(text) if word not in stop_words and len(word) < 15]).strip())
    return df

def soft_preprocess(df):
    df.iloc[:,1]=df.iloc[:,1].swifter.apply(ftfy.fix_text)
    df.iloc[:,1]=df.iloc[:,1].swifter.apply(clean_text)
    df.iloc[:,1]=df.iloc[:,1].swifter.apply(lambda x: x.replace('"', "'").replace("\n", " "))
    df.iloc[:,1]=df.iloc[:,1].swifter.apply(fix_contractions)
    return df

def df_to_hedwig_tsv(df, dsname, outfilename, num_labels_in_col, preprocess, is_twitter_process=True,
                     label_cols=[0], text_col=1):
    def to_tsv(outfpath, labels, texts):
        with open(outfpath, 'w', newline='') as tsvfile:
            writer = csv.writer(tsvfile, delimiter='\t')
            for label, text in zip(labels, texts):
                writer.writerow([label, text])
    df = preprocess(df)
    if is_twitter_process:
        df.iloc[:,1]=df.iloc[:,1].swifter.apply(tweet_preprocessing)
        
    df.iloc[:,0] = df.swifter.apply(lambda row: ''.join([str(lbl) for lbl in row[label_cols]]), axis=1)
    df = df.iloc[:,[0, 1]]
    df.iloc[:,0]=df.iloc[:,0].astype('str')
    df.iloc[:,0]=df.iloc[:,0].swifter.apply(
        lambda x: x if len(x) == num_labels_in_col else ''.join(
            ['0' for i in range(num_labels_in_col-len(x))]
        )+str(x)
    )
    dspath=PATH_TO_DATASETS/dsname
    outfpath = dspath/outfilename
    df = df.sample(frac=1.0)
    to_tsv(outfpath, df.iloc[:,0].tolist(), df.iloc[:,1].tolist())
    return df

def main(args):
    df = pd.read_csv(PATH_TO_DATASETS/args.dataset_name/args.dev)
    df_to_hedwig_tsv(df, dsname=args.dataset_name, outfilename='dev.tsv', num_labels_in_col=args.num_labels_in_col,preprocess=soft_preprocess, is_twiter_process=True)
    df = pd.read_csv(PATH_TO_DATASETS/dataset_name/args.test)
    df_to_hedwig_tsv(df, dsname=args.dataset_name, outfilename='test.tsv', num_labels_in_col=args.num_labels_in_col,preprocess=soft_preprocess, is_twiter_process=True)
    df = pd.read_csv(PATH_TO_DATASETS/dataset_name/args.train)
    df_to_hedwig_tsv(df, dsname=args.dataset_name, outfilename='train.tsv', num_labels_in_col=args.num_labels_in_col,preprocess=soft_preprocess, is_twiter_process=True)


if __name__== "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset-name", dest="dataset_name",
                        help="name of the dataset being created", metavar="DATASET_NAME")
    parser.add_argument("-n", "--num-labels", dest="num_labels_in_col",
                        help="number of labels contained in the column in case there is one label column (could be either 1 or labels could be a string stored in a column)",
                        metavar="NUM_LABELS_IN_COL")
    parser.add_argument("-x", "--train", dest="train",
                        help="name of the data file to be converted to train.tsv", metavar="TRAIN")
    parser.add_argument("-v", "--dev", dest="dev",
                        help="name of the data file to be converted to dev.tsv", metavar="DEV")
    parser.add_argument("-t", "--test", dest="test",
                        help="name of the data file to be converted to test.tsv", metavar="TEST")

    main(parser.parse_args())

