





#############################################################################
################ DREAMACHINE DATASET CONFIGURATION ##########################
#############################################################################

class DreamachineConfig:
    def __init__(self):
        from nltk.corpus import stopwords
        self.reduced_custom_stopwords = {'thank', 'thanks', 'thank you'}
        self.stop_words = set(stopwords.words('english'))
        self.extended_stop_words = self.stop_words.union(self.reduced_custom_stopwords)
        
        # Dataset specific configurations
        self.name = "dreamachine"
        self.transformer_model = "all-mpnet-base-v2"
        self.ngram_range = (1, 2)
        self.max_df = 0.95
        self.min_df = 3
        self.top_n_words = 10



    #############################################################################
    ################ HYPERPARAMETERS ############################################
    #############################################################################


    def get_params(self, condition, reduced=False):
        return self._get_reduced_params(condition) if reduced else self._get_full_params(condition)


    def _get_full_params(condition):
        if condition == 'HS':
            return {'umap_params': {
                    'n_components': list(range(3, 21)), 
                    'n_neighbors': [5,10,15,20,25],
                    'min_dist': [0.0,0.01,0.025,0.05], 
                },
                'hdbscan_params': {
                    'min_cluster_size': [5,10], 
                    'min_samples': [5,10],
                }}
        elif condition == 'DL':
            return {'umap_params': {
                    'n_components': list(range(3, 21)), 
                    'n_neighbors': [5,10,15,20,25],
                    'min_dist': [0.0,0.01,0.025,0.05], 
                },
                'hdbscan_params': {
                    'min_cluster_size': [5,10], 
                    'min_samples': [5,10],
                }}
        else:
            return {'umap_params': {
                    'n_components': list(range(3, 21)),
                    'n_neighbors': [10,15,20,25,30,35],
                    'min_dist': [0.01,0.025,0.05],
                },
                'hdbscan_params': {
                    'min_cluster_size': [10,20,30,40,50],
                    'min_samples': [None,10],
                }}


    def _get_reduced_params(condition):
        if condition == 'HS':
            return {'umap_params': {
                    'n_components': list(range(3, 10)),
                    'n_neighbors': [10],
                    'min_dist': [0.01],
                },
                'hdbscan_params': {
                    'min_cluster_size': [10],
                    'min_samples': [5],
                }}
        elif condition == 'DL':
            return {'umap_params': {
                    'n_components': list(range(3, 10)),
                    'n_neighbors': [10],
                    'min_dist': [0.01],
                },
                'hdbscan_params': {
                    'min_cluster_size': [10],
                    'min_samples': [5],
                }}
        else:
            return {'umap_params': {
                    'n_components': [10],
                    'n_neighbors': [20],
                    'min_dist': [0.01],
                },
                'hdbscan_params': {
                    'min_cluster_size': [10],
                    'min_samples': [10],
                }}
        

# create instance of the config class
config = DreamachineConfig()