from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap.umap_ import UMAP
from hdbscan import HDBSCAN


from src.utils import calculate_coherence, get_params_grid


def setup_model(umap_model, hdbscan_model, embedding_model, vectorizer_model, top_n_words=10, nr_topics="auto"):
    return BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        top_n_words=top_n_words, #default is 10
        nr_topics=nr_topics,
        language='english',
        calculate_probabilities=True,
        verbose=True
    )


def setup_umap(n_neighbors, n_components, min_dist, random_seed=42):
    return UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        metric='cosine',
        random_state=random_seed
    )

def setup_hdbscan(min_cluster_size, min_samples):
    return HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        gen_min_span_tree=True,
        prediction_data=True
    )


def run_bertopic(data, vectorizer_model, embedding_model, n_neighbors, n_components, min_dist, min_cluster_size, min_samples, top_n_words=10, nr_topics="auto"):
    """Run BERTopic model with given parameters."""
    umap_model = setup_umap(n_neighbors, n_components, min_dist)
    hdbscan_model = setup_hdbscan(min_cluster_size, min_samples)
    
    model = setup_model(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        top_n_words=top_n_words,
        nr_topics=nr_topics
    )

    topics, _ = model.fit_transform(data)
    coherence_score, coherence_score_umass = calculate_coherence(model, data)
    
    return model, topics, coherence_score, coherence_score_umass