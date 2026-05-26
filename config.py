"""
Configuration module containing central directory setups, file paths, and hyperparameters.
"""
import os

# Central Data Directory Configuration
DRIVE_DATA_DIR = '/content/drive/MyDrive/Dset/amazon DS/student_resource'
LOCAL_DATA_DIR = './data'

DATA_DIR = DRIVE_DATA_DIR if os.path.exists(DRIVE_DATA_DIR) else LOCAL_DATA_DIR

DATASET_DIR = os.path.join(DATA_DIR, 'dataset')
IMAGE_DIR = os.path.join(DATA_DIR, 'images')
EMB_DIR = os.path.join(DATA_DIR, 'embeddings')

# Data File Paths
TRAIN_CSV = os.path.join(DATASET_DIR, 'train.csv')
TEST_CSV = os.path.join(DATASET_DIR, 'sample_test.csv')
SAMPLE_OUT_CSV = os.path.join(DATASET_DIR, 'sample_test_out.csv')

# Hyperparameters
RANDOM_STATE = 42
NUM_CV_SPLITS = 5
TEXT_EMBEDDING_MODEL = 'paraphrase-MiniLM-L6-v2'
IMAGE_EMBEDDING_MODEL = 'google/vit-base-patch16-224'
