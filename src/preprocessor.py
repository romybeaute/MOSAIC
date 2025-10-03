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


def preproc(df_reports,sentences=True,min_words=2):
    #divide in sentences if needed
    if sentences:
        df_reports = split_sentences(df_reports)[0]
    print(f"\nSuccessfully loaded and processed {len(df_reports)} sentences.")

    # #remove sentences defined as too short
    # for i, sentence in enumerate(df_reports):
    #     if len(sentence.split()) < min_words:
    #         print(sentence)

    # #print the amount of sentences that have less than min_words words
    # short_sentences = [sentence for sentence in df_reports if len(sentence.split()) < min_words]
    # print(f"\nThere are {len(short_sentences)} sentences with less than {min_words} words.\n")

    # Remove sentences with less than min_words
    df_reports = [sentence for sentence in df_reports if len(sentence.split()) >= min_words]
    print(f"After removing short sentences, {len(df_reports)} sentences remain.")

    # Remove duplicate sentences if any
    seen = set()
    df_reports = [s for s in df_reports if not (s in seen or seen.add(s))]
    print(f"After removing duplicates, {len(df_reports)} remain.")
    return df_reports


def clean_text(text):
    # to lowercase
    text = text.lower()
    # Remv special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # rmv extra whitespace
    text = ' '.join(text.split())
    return text





