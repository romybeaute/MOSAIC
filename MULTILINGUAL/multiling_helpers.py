
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
from sudachipy import tokenizer
from sudachipy import dictionary



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
    
    # any class that inherits from LanguageProcessor MUST implement its own version of setup_stopwords and split_sentences, as may differ wrt language analysed!
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
        
    # def split_sentences(self, reflections):
    #     '''
    #     uses NLTK's sentence tokenizer specifically configured for eng
    #     '''
    #     return [sent for text in reflections 
    #             for sent in nltk.sent_tokenize(text, language='english')]

    def split_sentences(self, reflections):
        '''
        uses NLTK's sentence tokenizer specifically configured for eng
        '''
        sentences = []
        doc_map = []
        
        for doc_idx, text in enumerate(reflections):
            if isinstance(text, str):
                doc_sentences = nltk.sent_tokenize(text, language='english')
                sentences.extend(doc_sentences)
                doc_map.extend([doc_idx] * len(doc_sentences))
        
        return sentences, doc_map

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
        
    # def split_sentences(self, reflections):
    #     '''
    #     uses same NLTK's sentence tokenizer but configured for french language rules
    #     '''
    #     return [sent for text in reflections 
    #             for sent in nltk.sent_tokenize(text, language='french')]

    def split_sentences(self, reflections):
        '''
        uses NLTK's sentence tokenizer specifically configured for eng
        '''
        sentences = []
        doc_map = []
        
        for doc_idx, text in enumerate(reflections):
            if isinstance(text, str):
                doc_sentences = nltk.sent_tokenize(text, language='french')
                sentences.extend(doc_sentences)
                doc_map.extend([doc_idx] * len(doc_sentences))
        
        return sentences, doc_map

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
        
    # def split_sentences(self, reflections):
    #     '''
    #     uses same NLTK's sentence tokenizer but configured for portuguese language rules
    #     '''
    #     return [sent for text in reflections 
    #             for sent in nltk.sent_tokenize(text, language='portuguese')]
    def split_sentences(self, reflections):
        '''
        uses NLTK's sentence tokenizer specifically configured for eng
        '''
        sentences = []
        doc_map = []
        
        for doc_idx, text in enumerate(reflections):
            if isinstance(text, str):
                doc_sentences = nltk.sent_tokenize(text, language='portuguese')
                sentences.extend(doc_sentences)
                doc_map.extend([doc_idx] * len(doc_sentences))
        
        return sentences, doc_map


class JapaneseProcessor(LanguageProcessor):
    def __init__(self):
        super().__init__('japanese')
        self.setup_tokenizer()
        self.setup_stopwords()
        self.sentence_endings = {'。', '．', '.', '！', '!', '？', '?', '\n'}

    def setup_tokenizer(self):
        """Initialise Sudachi tokenizer with full dictionary"""
        try:
            self.tokenizer_obj = dictionary.Dictionary(dict="full").create()
            self.mode = tokenizer.Tokenizer.SplitMode.B  # Balanced mode
        except ImportError:
            print("Error: Install Sudachi - pip install sudachipy sudachidict_full")
            self.tokenizer_obj = None

    def setup_stopwords(self):
        """Combine NLTK, custom, and domain-specific stopwords"""
        # Base NLTK stopwords
        self.stopwords = set()
        
        ja_stopwords_inner_speech = ["は", "が", "を", "に", "で", "と", "も", "へ"] #basic particles
            # # Function words (particles, conjunctions, auxiliaries)
            # "が", "の", "に", "を", "で", "と", "も", "へ", "や", "から", "まで",
            # "ば", "より", "か", "し", "せ", "た", "だ", "う", "つ", "な", "ら", "れ", "ん",
            # "そして", "しかし", "また", "ただし", "および", "おり", "おります", "では",
            # "ながら", "ので", "など", "のみ", "ため", "にて", "として", "とき", "とともに",
            # "と共に", "について", "において", "における", "により", "による", "に対して",
            # "に対する", "に関する", "その他", "その後", "さらに", "でも", "という", "といった",

            # # Demonstratives (except "this"/"I"-type pronouns)
            # "ここ", "そこ", "あそこ", "どこ", "これら", "それ", "あれ", "あの", "この", "その",

            # # Generic determiners / modifiers
            # "こと", "もの", "こと", "ため", "ところ", "ものの", "よう", "といった", "たち"


        

        # Custom stopwords for Japanese inner speech analysis
        custom_stopwords = [
            # Greetings/Thanks
            "ありがとう", "よろしく", "お願いします", "失礼します", "すみません",
            "ありがとうございました", "ありがとうございます",  # Formal thanks
            "感謝", "恐縮",  # "Gratitude", "Humble apology"

            # Research Feedback Phrases
            "面白い", "興味深い", "楽しかった",  # "Interesting", "Fascinating", "Enjoyable"
            "勉強になりました", "参考になりました",  # "Was educational", "Was helpful"
            "応援してます", "頑張ってください",  # "Rooting for you", "Do your best"
            "素晴らしい", "期待してます",  # "Wonderful", "Looking forward to"
            "成果", "成功",  # "Results", "Success"

            # Apologetic/Politeness
            "申し訳ありません", "失礼いたしました",  # Formal apologies
            "お手数", "ご容赦",  # "Trouble", "Forgiveness"
            "恐れ入ります", "お邪魔します",  # Polite phrases

            # Research Artifacts
            "アンケート", "調査", "研究", "質問", "回答",
            "論文", "実験", "分析",  # "Paper", "Experiment", "Analysis"

            # Honorifics
            "です", "ます", "ました", "ございます",
            "させていただきます", "くださいまして"  # Humble forms
        ]

        self.stopwords.update(ja_stopwords_inner_speech)
        self.stopwords.update(custom_stopwords)

    def preprocess_text(self, text):
        """Full preprocessing pipeline"""
        if not self.tokenizer_obj:
            return text  # Fallback if Sudachi not installed
            
        # Basic cleaning from parent class
        text = super().preprocess_text(text)
        
        # Sudachi tokenization with POS filtering
        tokens = []
        for m in self.tokenizer_obj.tokenize(text, self.mode):
            pos = m.part_of_speech()[0]
            if pos != '助詞':
                tokens.append(m.dictionary_form())
        
        # Stopword removal
        tokens = [t for t in tokens if t not in self.stopwords]
        
        return ' '.join(tokens)
    



    def split_sentences(self, reflections):
        '''
        uses ja-sentence tokenizer for Japanese sentence splitting
        and returns document mapping information
        '''
        try:
            from ja_sentence.tokenizer import tokenize
            sentences = []
            doc_map = []  # track which document each sentence belongs to
            
            for doc_idx, text in enumerate(reflections):
                if isinstance(text, str):  # check if text is string
                    doc_sentences = tokenize(text)
                    sentences.extend(doc_sentences)
                    doc_map.extend([doc_idx] * len(doc_sentences))
            
            return sentences, doc_map
        except ImportError:
            print("Please install ja-sentence: pip install ja-sentence")
            return reflections, list(range(len(reflections)))
    



class TopicModeler:
    """Class for topic modeling in multiple languages"""
    def __init__(self, language_processor, use_stopwords=True):
        """
        Initialize topic modeler with specific language processor
        Args:
            language_processor: Instance of LanguageProcessor for specific language
            use_stopwords: Whether to use stopwords in vectorizer (default to True)
        """
        self.language_processor = language_processor
        self.use_stopwords = use_stopwords
        self.embedding_model = SentenceTransformer(
            self.language_processor.sentence_transformer_model
        )
        
    def prepare_vectorizer(self, ngram_range=(1,2), max_df=0.9, min_df=2):
        stopwords = list(self.language_processor.stopwords) if self.use_stopwords else None
        return CountVectorizer(
            ngram_range=ngram_range,
            stop_words=stopwords,
            max_df=max_df,
            min_df=min_df
        )
        
    def process_data(self, data, split_sentences=True):
        if split_sentences:
            return self.language_processor.split_sentences(data)
        return data

