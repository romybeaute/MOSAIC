from bertopic import BERTopic
from umap.umap_ import UMAP
from hdbscan import HDBSCAN
from bertopic.representation import KeyBERTInspired

from mosaic.utils import calculate_coherence,calculate_embedding_coherence


def setup_model(umap_model, hdbscan_model, embedding_model, vectorizer_model, representation_model, top_n_words, nr_topics="auto"):
    return BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        top_n_words=top_n_words, #default is 10
        nr_topics=nr_topics,
        representation_model=representation_model,
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
        gen_min_span_tree=False,
        prediction_data=True
    )


def run_bertopic(data, embeddings, vectorizer_model, embedding_model, n_neighbors, n_components, min_dist, min_cluster_size, min_samples, top_n_words, nr_topics="auto",representation_method=None,random_seed=42):
    """Run BERTopic model with given parameters."""
    umap_model = setup_umap(n_neighbors, n_components, min_dist,random_seed=random_seed)
    hdbscan_model = setup_hdbscan(min_cluster_size, min_samples)

    if representation_method == 'keybert':
        representation_model = KeyBERTInspired()
    else: # Default to c-TF-IDF
        representation_model = None

    
    model = setup_model(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        top_n_words=top_n_words,
        representation_model=representation_model,
        nr_topics=nr_topics
    )

    topics, _ = model.fit_transform(data,embeddings)
    coherence_score = calculate_coherence(model, data)
    s_embedding_coherence = calculate_embedding_coherence(model, data, embeddings)
    # inter_topic_sim = calculate_inter_topic_similarity(model, embeddings)
    
    return model, topics, coherence_score, s_embedding_coherence