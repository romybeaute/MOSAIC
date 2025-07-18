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
    """
    Split list of texts into sentences and track which sentence belongs to which document.
    
    Parameters:
    -----------
    reflections : list
        A list of strings (documents/reflections)
        
    Returns:
    --------
    tuple
        (sentences, doc_map) where:
        - sentences is a list of all sentences from all documents
        - doc_map is a list of indices indicating which document each sentence belongs to
    """
    tokenizer = PunktSentenceTokenizer()
    sentences = []
    doc_map = [] 
    
    for doc_idx, reflection in enumerate(reflections):
        doc_sentences = tokenizer.tokenize(reflection)
        sentences.extend(doc_sentences)
        doc_map.extend([doc_idx] * len(doc_sentences))
    
    return sentences, doc_map


def clean_text(text):
    # to lowercase
    text = text.lower()
    # Remv special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # rmv extra whitespace
    text = ' '.join(text.split())
    return text





