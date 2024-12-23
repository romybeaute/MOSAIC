from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap.umap_ import UMAP
from hdbscan import HDBSCAN

def get_model_params(condition, len_dataset):
    """Get hyperparameters based on condition."""
    params = {
        'HS': {
            'umap_params': {'n_components': range(3, 21)},
            'hdbscan_params': {'min_cluster_size': [5, 10]}
        },
        'DL': {
            'umap_params': {'n_components': range(3, 21)},
            'hdbscan_params': {'min_cluster_size': [5, 10]}
        }
    }
    return params.get(condition)



def setup_topic_model(embedding_model, umap_model, hdbscan_model, top_n_words):
    """Configure BERTopic model with parameters."""
    return BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        top_n_words=top_n_words,
        language='english'
    )