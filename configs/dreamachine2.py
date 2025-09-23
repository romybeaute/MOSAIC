





#############################################################################
################ DREAMACHINE DATASET CONFIGURATION ##########################
#############################################################################

class DreamachineConfig:
    def __init__(self):
        from nltk.corpus import stopwords
        self.reduced_custom_stopwords = {}#{'felt','like','felt like','feel','experience','experienced'}#{'thank', 'thanks', 'thank you','Thank','felt','felt like','experience','experienced'}
        self.stop_words = set(stopwords.words('english'))
        self.extended_stop_words = self.stop_words.union(self.reduced_custom_stopwords)
        
        # Dataset specific configurations
        self.name = "dreamachine2.0"
        self.transformer_model = "Qwen/Qwen3-Embedding-0.6B"
        self.ngram_range = (1, 3)
        self.max_df = 0.95
        self.min_df = 2
        self.top_n_words = 15

    def get_default_params(self, condition):
        """
        Returns a single set of default hyperparameters based on the condition.
        These are derived from the optimal values found in the original paper.
        """
        if condition == 'HS':
            return {
                'n_neighbors': 15,
                'n_components': 10,
                'min_dist': 0.0,
                'min_cluster_size': 10,
                'min_samples': 5,
                'top_n_words': self.top_n_words  # Using a good default for keyword extraction
            }
        elif condition == 'DL':
            return {
                'n_neighbors': 10,
                'n_components': 5,
                'min_dist': 0.0,
                'min_cluster_size': 5,
                'min_samples': 3,
                'top_n_words': self.top_n_words
            }
        else: # A sensible generic default for any other condition
            return {
                'n_neighbors': 15,
                'n_components': 5,
                'min_dist': 0.0,
                'min_cluster_size': 10,
                'min_samples': 5,
                'top_n_words': self.top_n_words
            }




    #############################################################################
    ################ HYPERPARAMETERS ############################################
    #############################################################################


    def get_params(self, condition, reduced=False):
        return self._get_reduced_params(condition) if reduced else self._get_full_params(condition)


    def _get_full_params(self,condition):
        if condition == 'HS':
            return {'umap_params': {
                    'n_components': [5,10,15], 
                    'n_neighbors': [10,15,20,25],
                    'min_dist': [0.0,0.1], 
                },
                'hdbscan_params': {
                    'min_cluster_size': [5,10,15], 
                    'min_samples': [5,10],
                }}
        elif condition == 'DL':
            return {'umap_params': {
                    'n_components': list(range(2, 10)), 
                    'n_neighbors': [5,10,15],
                    'min_dist': [0.0,0.1], 
                },
                'hdbscan_params': {
                    'min_cluster_size': [5,10], 
                    'min_samples': [3,5],
                }}
        else:
            return {'umap_params': {
                    'n_components': list(range(3, 21)),
                    'n_neighbors': [10,15,20,25,30,35],
                    'min_dist': [0.0,0.1],
                },
                'hdbscan_params': {
                    'min_cluster_size': [10,20,30,40,50],
                    'min_samples': [None,10],
                }}


    def _get_reduced_params(self,condition):
        if condition == 'HS':
            return {'umap_params': {
                    'n_components': [10],
                    'n_neighbors': [15],
                    'min_dist': [0.0],
                },
                'hdbscan_params': {
                    'min_cluster_size': [10],
                    'min_samples': [5],
                }}
        elif condition == 'DL':
            return {'umap_params': {
                    'n_components': [5],
                    'n_neighbors': [10],
                    'min_dist': [0.0],
                },
                'hdbscan_params': {
                    'min_cluster_size': [5],
                    'min_samples': [3],
                }}
        else:
            return {'umap_params': {
                    'n_components': [10],
                    'n_neighbors': [20],
                    'min_dist': [0.0],
                },
                'hdbscan_params': {
                    'min_cluster_size': [10],
                    'min_samples': [10],
                }}
        

# create instance of the config class
config = DreamachineConfig()