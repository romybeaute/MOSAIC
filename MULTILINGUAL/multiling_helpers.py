
#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : multiling_helpers.py
# description     : Define helpers functions for multilingual pipeline
# author          : Romy, Beauté (r.beaut@sussex.ac.uk)
# date            : 03-12-2024
# ==============================================================================

import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer

#japanese dependencies 
from ja_sentence.tokenizer import tokenize

class LanguageProcessor:
    """Base class for language processing"""
    def __init__(self, language):
        if language not in ["english", "french", "japanese","portuguese"]:
            raise ValueError(f"Unsupported language: {language}")
        self.language = language
        self.stopwords = set()
        self.sentence_transformer_model = self._get_transformer_model()
        
    def _get_transformer_model(self):
        """
        Get appropriate transformer model for language
        Check MTEB Leaderbord: https://huggingface.co/spaces/mteb/leaderboard
        """
        model_map = {
            'english': "all-mpnet-base-v2",
            'french': "bge-multilingual-gemma2",#"paraphrase-multilingual-mpnet-base-v2",
            'japanese': "bge-multilingual-gemma2", #"paraphrase-multilingual-mpnet-base-v2"
            'portuguese': "bge-multilingual-gemma2" #"paraphrase-multilingual-mpnet-base-v2"
        }
        return model_map.get(self.language, "paraphrase-multilingual-mpnet-base-v2") # default to paraphrase-multilingual-mpnet-base-v2
    
    # any class that inherits from LanguageProcessor MUST implement its own version of setup_stopwords and split_sentences !
    def setup_stopwords(self):
        """to be implemented by child classes"""
        raise NotImplementedError
    
    def preprocess_text(self, text):
        """Basic text preprocessing for all languages"""
        if not isinstance(text, str):
            return text
            
        # Remove extra whitespace (including multiple spaces, tabs, newlines)
        text = ' '.join(text.split())
        
        # Remove multiple punctuation marks (like "!!!" or "???")
        text = re.sub(r'([!?.]){2,}', r'\1', text)
        
        # Standardize quotes
        text = text.replace('"', '"').replace('"', '"')
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # # if wabt to remove numbers with context (keep if part of word)
        # text = re.sub(r'\s\d+\s', ' ', text)
        
        return text.strip()

    def split_sentences(self, text):
        """to be implemented by child classes"""
        raise NotImplementedError
    

    def print_info(self):
        """Print information about the processor configuration"""
        print(f"Language: {self.language}")
        print(f"Transformer model: {self.sentence_transformer_model}")
        print(f"Number of stopwords: {len(self.stopwords)}")
        if hasattr(self, 'sentence_endings'):
            print(f"Sentence endings: {self.sentence_endings}")


######  defione children classes for each language separately ######

class EnglishProcessor(LanguageProcessor):
    def __init__(self):
        super().__init__('english')
        self.setup_stopwords()
        
    def setup_stopwords(self):
        nltk.download('stopwords')
        self.stopwords = set(stopwords.words('english'))

    def preprocess_text(self, text):
        # apply basic preprocessing as defined in parent class 
        text = super().preprocess_text(text)
        # add more specific preprocessing for english (TO-DO later)
        return text
        
    def split_sentences(self, reflections):
        '''
        uses NLTK's sentence tokenizer specifically configured for eng
        '''
        return [sent for text in reflections 
                for sent in nltk.sent_tokenize(text, language='english')]

class FrenchProcessor(LanguageProcessor):
    def __init__(self):
        super().__init__('french')
        self.setup_stopwords()
        
    def setup_stopwords(self):
        nltk.download('stopwords')
        self.stopwords = set(stopwords.words('french'))
    
    def preprocess_text(self, text):
        # apply basic preprocessing as defined in parent class 
        text = super().preprocess_text(text)
        # add more specific preprocessing for french (TO-DO later)
        return text
        
    def split_sentences(self, reflections):
        '''
        uses same NLTK's sentence tokenizer but configured for french language rules
        '''
        return [sent for text in reflections 
                for sent in nltk.sent_tokenize(text, language='french')]

class PortugueseProcessor(LanguageProcessor):
    def __init__(self):
        super().__init__('portuguese')
        self.setup_stopwords()
        
    def setup_stopwords(self):
        nltk.download('stopwords')
        self.stopwords = set(stopwords.words('portuguese'))

    def preprocess_text(self, text):
        # apply basic preprocessing as defined in parent class 
        text = super().preprocess_text(text)
        # add more specific preprocessing for portuguese (TO-DO later)
        return text
        
    def split_sentences(self, reflections):
        '''
        uses same NLTK's sentence tokenizer but configured for portuguese language rules
        '''
        return [sent for text in reflections 
                for sent in nltk.sent_tokenize(text, language='portuguese')]


class JapaneseProcessor(LanguageProcessor):
    def __init__(self):
        super().__init__('japanese')
        self.setup_stopwords()
        self.sentence_endings = set(['。', '．', '.', '！', '!', '？', '?', '\n']) # add suppl japanese sentence endings (not added to piepline yet !! => need to see if we need them)
        
    def setup_stopwords(self):
        try:
            import MeCab
            self.mecab = MeCab.Tagger("-Owakati")
            # complete with add common Japanese stopwords
            self.stopwords = set([
                # Original particles and markers
                'の', 'に', 'は', 'を', 'た', 'が', 'で', 'て', 'と',
                'です', 'ます', 'した', 'ない', 'する', 'ある',
                'これ', 'それ', 'あれ', 'この', 'その', 'あの',
                '私', '僕', '俺', '自分',
                
                # Thank you expressions
                'ありがとう',
                'ありがとうございます',
                'ありがとうございました',
                'どうも',
                'どうもありがとう',
                'どうもありがとうございます',
                'どうもありがとうございました',
                'サンキュー',
                'サンクス',
                'お礼',
                'かたじけない',
                'Thank you',
                'Thanks'
            ])
        except ImportError:
            print("Please install MeCab: pip install mecab-python3 unidic-lite")
    
    def preprocess_text(self, text):
        # apply basic preprocessing as defined in parent class 
        text = super().preprocess_text(text)
        # add more specific preprocessing for japanese (TO-DO later)
        return text
            
    def split_sentences(self, reflections):
        '''
        uses ja-sentence tokenizer for Japanese sentence splitting
        and additional custom splitting for periods
        '''
        try:
            from ja_sentence.tokenizer import tokenize
            sentences = []
            for text in reflections:
                if isinstance(text, str):  # Check if text is string
                    paragraphs = text.split('\n')
                    for paragraph in paragraphs:
                        if paragraph.strip():  # Only process non-empty paragraphs
                            # get initial splits from ja-sentence
                            initial_splits = tokenize(paragraph)
                            for split in initial_splits:
                                # Create a regex pattern from all sentence endings
                                pattern = f"[{''.join(self.sentence_endings)}]"
                                further_splits = [s.strip() for s in re.split(pattern, split) if s.strip()]
                                sentences.extend(further_splits)
                                # sentences.extend(split.split('。'))
                            
            return sentences
        except ImportError:
            print("Please install ja-sentence: pip install ja-sentence")
            return reflections

class TopicModeler:
    """Class for topic modeling in multiple languages"""
    def __init__(self, language_processor):
        """
        Initialize topic modeler with specific language processor
        Args:
            language_processor: Instance of LanguageProcessor for specific language
        """
        self.language_processor = language_processor
        self.embedding_model = SentenceTransformer(
            self.language_processor.sentence_transformer_model
        )
        
    def prepare_vectorizer(self, ngram_range=(1,2), max_df=0.9, min_df=2):
        return CountVectorizer(
            ngram_range=ngram_range,
            stop_words=list(self.language_processor.stopwords),
            max_df=max_df,
            min_df=min_df
        )
        
    def process_data(self, data, split_sentences=True):
        if split_sentences:
            return self.language_processor.split_sentences(data)
        return data

