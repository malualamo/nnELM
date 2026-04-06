from os import path
# Configuration constants
# All paths are relative to root
CONFIG_DIR    = path.join('Model', 'configs')
DATASETS_PATH = 'Datasets'
RESULTS_PATH  = 'Results'
SALIENCY_PATH = path.join('Results', 'saliency_maps')
TARGET_SIMILARITY_PATH = path.join('Results','target_similarity_maps')

SIGMA      = [[4000, 0], [0, 2600]]
IMAGE_SIZE = (768, 1024)