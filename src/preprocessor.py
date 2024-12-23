#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : preprocessor.py
# description     : Define helpers functions
#                   Condition can be either HS,DL or HW
# author          : Romy, Beauté (r.beaut@sussex.ac.uk)
# date            : 2024-07-25
# ==============================================================================

import pandas as pd
import numpy as np
from tqdm import tqdm
import re


from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WordPunctTokenizer

import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import PunktSentenceTokenizer



#############################################################################
################ DATA PREPROC : SENTENCES ###################################
#############################################################################






def split_sentences(reflections):
    tokenizer = PunktSentenceTokenizer()
    sentences = []
    for reflection in reflections:
        sentences += tokenizer.tokenize(reflection)
    return sentences


def clean_text(text):
    # to lowercase
    text = text.lower()
    # Remv special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # rmv extra whitespace
    text = ' '.join(text.split())
    return text





