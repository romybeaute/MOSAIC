import pandas as pd
from tqdm import tqdm
from scipy.sparse import csr_matrix
from llama_cpp import Llama
from typing import Mapping, List, Tuple, Any, Union, Callable
from bertopic.representation._base import BaseRepresentation
from bertopic.representation._utils import truncate_document, validate_truncate_document_parameters


DEFAULT_PHENO_PROMPT = """
This is a list of sentences where each collection of sentences describe a topic. After each collection of sentences, the name of the topic they represent is mentioned as a short-highly-descriptive scientific title.
---
Topic:
Sample texts from this topic:
- I experienced a sense of oneness with all things
- I felt unity with the universe or everything.
- I felt dissolved boundaries between myself and the outside world.
- I experienced a loss of ego or self-boundaries.
- I had a feeling of interconnectedness with others and nature.


Keywords: Oneness, Unity, Ego dissolution, Interconnectedness, Wholeness, Boundaries dissolved, Cosmic connectedness, Self-transcendence, Universal merging, Totality
Topic name: Experience of Unity
---
Topic:
Sample texts from this topic:
-I saw complex, rapidly changing visual scenes with my eyes closed.
- I experienced vivid, dream-like images or visions.
- My imagination produced detailed and elaborate mental pictures.
- I perceived symbolic or meaningful images spontaneously.
- I felt as if I was watching a movie of intense and complex images inside my mind.

Keywords:
Complex imagery, Vivid visions, Dream-like images, Visual scenes, Mental pictures, Symbolic images, Imagination, Elaborate visuals, Inner movie, Visual richness

Topic name:
Complex Imagery
---
Topic:
Sample sentences from this topic:
[SENTENCES]

Keywords:
- c-TF-IDF: [KEYWORDS]
- KeyBERT: [KEYBERT_KEYWORDS]
- MMR: [MMR_KEYWORDS]

Topic name:"""

DEFAULT_PHENO_SYSTEM_PROMPT = "You are a research assistant. Your task is to summarize the core subjective experience from the provided texts into a concise, scientific title."
DEFAULT_SYSTEM_PROMPT = "You are an assistant that extracts high-level topics from texts." #the same as the LlamaCPP library (to match)


class PhenoLabeler(BaseRepresentation):
    """A custom representation model for generating phenomenological labels using multiple keyword sets and a Llama 3 model.

    Arguments:
        model: Either a string pointing towards a local LLM or a
                `llama_cpp.Llama` object.
        prompt: The prompt to be used in the model. If no prompt is given,
                `self.default_prompt_` is used instead.
                Use `"[KEYWORDS]"` and `"[SENTENCES]"` in the prompt
                to decide where the keywords and sentences need to be
                inserted.
        system_prompt: The system prompt to be used in the model. If no system prompt is given,
                       `self.default_system_prompt_` is used instead.
        pipeline_kwargs: Kwargs that you can pass to the `llama_cpp.Llama`
                         when it is called such as `max_tokens` to be generated.
        nr_docs: The number of sentences to pass to OpenAI if a prompt
                 with the `["SENTENCES"]` tag is used.
        diversity: The diversity of sentences to pass to OpenAI.
                   Accepts values between 0 and 1. A higher
                   values results in passing more diverse sentences
                   whereas lower values passes more similar sentences.
        doc_length: The maximum length of each document. If a document is longer,
                    it will be truncated. If None, the entire document is passed.
        tokenizer: The tokenizer used to calculate to split the document into segments
                   used to count the length of a document.
                       * If tokenizer is 'char', then the document is split up
                         into characters which are counted to adhere to `doc_length`
                       * If tokenizer is 'whitespace', the the document is split up
                         into words separated by whitespaces. These words are counted
                         and truncated depending on `doc_length`
                       * If tokenizer is 'vectorizer', then the internal CountVectorizer
                         is used to tokenize the document. These tokens are counted
                         and truncated depending on `doc_length`
                       * If tokenizer is a callable, then that callable is used to tokenize
                         the document. These tokens are counted and truncated depending
                         on `doc_length`

    Usage:

    To use a llama.cpp, first download the LLM:

    ```bash
    wget https://huggingface.co/TheBloke/zephyr-7B-alpha-GGUF/resolve/main/zephyr-7b-alpha.Q4_K_M.gguf
    ```

    Then, we can now use the model the model with BERTopic in just a couple of lines:

    ```python
    from bertopic import BERTopic
    from bertopic.representation import LlamaCPP

    # Use llama.cpp to load in a 4-bit quantized version of Zephyr 7B Alpha
    representation_model = LlamaCPP("zephyr-7b-alpha.Q4_K_M.gguf")

    # Create our BERTopic model
    topic_model = BERTopic(representation_model=representation_model, verbose=True)
    ```

    If you want to have more control over the LLMs parameters, you can run it like so:

    ```python
    from bertopic import BERTopic
    from bertopic.representation import LlamaCPP
    from llama_cpp import Llama

    # Use llama.cpp to load in a 4-bit quantized version of Zephyr 7B Alpha
    llm = Llama(model_path="zephyr-7b-alpha.Q4_K_M.gguf", n_gpu_layers=-1, n_ctx=4096, stop="Q:")
    representation_model = LlamaCPP(llm)

    # Create our BERTopic model
    topic_model = BERTopic(representation_model=representation_model, verbose=True)
    ```
    """

    def __init__(
        self,
        model: Union[str, Llama],
        prompt: str = None,
        system_prompt: str = None,
        pipeline_kwargs: Mapping[str, Any] = {},
        nr_docs: int = 5,
        diversity: float = None,
        doc_length: int = None,
        tokenizer: Union[str, Callable] = None,
        verbose: bool = False
    ):
        if isinstance(model, str):
            self.model = Llama(model_path=model, n_gpu_layers=-1, stop=["\n", "Label:", "Topic:", "Keywords:","(Note:"], chat_format="ChatML")
        elif isinstance(model, Llama):
            self.model = model
        else:
            raise ValueError(
                "Make sure that the model that you"
                "pass is either a string referring to a"
                "local LLM or a ` llama_cpp.Llama` object."
            )
        self.prompt = prompt if prompt is not None else DEFAULT_PHENO_PROMPT
        self.system_prompt = system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT
        self.default_prompt_ = DEFAULT_PHENO_PROMPT
        self.default_system_prompt_ = DEFAULT_SYSTEM_PROMPT
        self.pipeline_kwargs = pipeline_kwargs
        self.nr_docs = nr_docs
        self.diversity = diversity
        self.doc_length = doc_length
        self.tokenizer = tokenizer
        validate_truncate_document_parameters(self.tokenizer, self.doc_length)
        self.verbose = verbose 

        self.prompts_ = []

    def extract_topics(
        self,
        topic_model,
        sentences: pd.DataFrame,
        c_tf_idf: csr_matrix,
        topics: Mapping[str, List[Tuple[str, float]]],
    ) -> Mapping[str, List[Tuple[str, float]]]:
        """Extract topic representations and return a single label.

        Arguments:
            topic_model: A BERTopic model
            sentences: Not used
            c_tf_idf: Not used
            topics: The candidate topics as calculated with c-TF-IDF

        Returns:
            updated_topics: Updated topic representations
        """
        topic_info_df = topic_model.get_topic_info()
        # Extract the top 4 representative sentences per topic
        repr_docs_mappings, _, _, _ = topic_model._extract_representative_docs(
            c_tf_idf, sentences, topics, 500, self.nr_docs, self.diversity
        )

        updated_topics = {}
        for topic, docs in tqdm(repr_docs_mappings.items(), disable=not topic_model.verbose):

            #Get the extra keywords set
            row = topic_info_df.loc[topic_info_df.Topic == topic].iloc[0]
            keybert_kws = ", ".join(row['KeyBERT'])
            mmr_kws = ", ".join(row['MMR'])

            # Prepare prompt
            truncated_docs = [truncate_document(topic_model, self.doc_length, self.tokenizer, doc) for doc in docs]
            prompt = self._create_prompt(truncated_docs, topic, topics,keybert_kws,mmr_kws)
            self.prompts_.append(prompt)


            # Extract result from generator and use that as label
            # topic_description = self.model(prompt, **self.pipeline_kwargs)["choices"]
            topic_description = self.model.create_chat_completion(
                messages=[{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}],
                **self.pipeline_kwargs,
            )
            label = topic_description["choices"][0]["message"]["content"].strip()
            updated_topics[topic] = [(label, 1)] + [("", 0) for _ in range(9)]

        return updated_topics

    def _create_prompt(self, docs, topic, topics,keybert_kws,mmr_kws):
        keywords = list(zip(*topics[topic]))[0] #gets the default c-tf-idf keywords

        # Use the Default Chat Prompt
        if self.prompt == self.default_prompt_:
            prompt = self.prompt.replace("[KEYWORDS]", ", ".join(keywords))
            prompt = prompt.replace("[KEYBERT_KEYWORDS]", keybert_kws)
            prompt = prompt.replace("[MMR_KEYWORDS]", mmr_kws)        
            prompt = self._replace_sentences(prompt, docs)

        # Use a custom prompt that leverages keywords, sentences or both using
        # custom tags, namely [KEYWORDS] and [SENTENCES] respectively
        else:
            prompt = self.prompt
            if "[KEYWORDS]" in prompt:
                prompt = prompt.replace("[KEYWORDS]", ", ".join(keywords))
            if "[KEYBERT_KEYWORDS]" in prompt:                          
                prompt = prompt.replace("[KEYBERT_KEYWORDS]", keybert_kws)
            if "[MMR_KEYWORDS]" in prompt:                          
                prompt = prompt.replace("[MMR_KEYWORDS]", mmr_kws)
            if "[SENTENCES]" in prompt:
                prompt = self._replace_sentences(prompt, docs)

        if self.verbose:
            print("\n" + "="*60)
            print(f"✅ FINAL PROMPT BEING SENT TO LLM FOR TOPIC: {topic}")
            print("="*60)
            print(prompt)
            print("="*60 + "\n")

        return prompt

    @staticmethod
    def _replace_sentences(prompt, docs):
        to_replace = ""
        for doc in docs:
            to_replace += f"- {doc}\n"
        prompt = prompt.replace("[SENTENCES]", to_replace)
        return prompt







from bertopic.representation import BaseRepresentation
import re

class MultiKeywordLLM(BaseRepresentation):
    """
    Other implementaition version of a custom representation model for generating labels with more keyword sets
    """
    def __init__(self, llm, prompt, pipeline_kwargs,nr_docs=10):
        super().__init__()
        self.llm = llm
        self.prompt = prompt
        self.pipeline_kwargs = pipeline_kwargs
        self.nr_docs = nr_docs 

    def extract_topics(self, topic_model, documents, c_tf_idf, topics):
        """
        This is the main function. For each topic, it will build a custom
        prompt and generate a new label.
        """
        # Get the topic info DataFrame that contains KeyBERT keywords
        topic_info_df = topic_model.get_topic_info()
        
        # Create the new labels
        new_labels = {}
        for index, row in topic_info_df.iterrows():
            topic_id = row['Topic']
            if topic_id == -1:
                continue

            # Get documents and all keyword sets for the current topic
            docs_to_use = topic_model.representative_docs_[topic_id][:self.nr_docs]
            docs = "\n".join(docs_to_use)

            ctfidf_kws = ", ".join([word for word, score in topic_model.get_topic(topic_id)])
            keybert_kws = ", ".join(row['KeyBERT'])
            
            # Fill in our detailed prompt
            filled_prompt = self.prompt.replace("[DOCUMENTS]", docs)
            filled_prompt = filled_prompt.replace("[KEYWORDS]", ctfidf_kws)
            filled_prompt = filled_prompt.replace("[KEYBERT_KEYWORDS]", keybert_kws)

            # Get the new label from the LLM
            response = self.llm(prompt=filled_prompt, **self.pipeline_kwargs)
            label = response['choices'][0]['text'].strip()
            new_labels[topic_id] = re.sub(r'^\d+\.\s*', '', label)
        
        return new_labels