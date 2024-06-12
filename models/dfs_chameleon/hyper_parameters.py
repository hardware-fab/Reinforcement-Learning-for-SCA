from . import state_space_parameters as ssp
import metaqnn.data_loader as data_loader
import numpy as np

MODEL_NAME = "dfs_chameleon_5" # 5 for aggregation
CIPHER = 'AES'

# Number of output neurons
NUM_CLASSES = 256  # Number of output neurons

# Input Size
INPUT_SIZE = 1_000

# Batch Queue parameters
TRAIN_BATCH_SIZE = 256  # Batch size for training (scaled linearly with number of gpus used)
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 183_580 # Number of training examples
NUM_ITER_PER_EPOCH_TRAIN = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / TRAIN_BATCH_SIZE
EVAL_BATCH_SIZE = TRAIN_BATCH_SIZE  # Batch size for validation
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 25_413 # Number of validation examples

MAX_EPOCHS = 50 # Max number of epochs to train model

# Training Parameters
OPTIMIZER = 'Adam'  # Optimizer (should be in caffe format string)
MAX_LR = 3e-3  # The max LR (scaled linearly with number of gpus used)

ATTACK_KEY_BYTE = 0

# Bulk data folder - where to save logs
BULK_ROOT = '../experiment/' # It must contains three sub-folders: graphs, qlearner_logs, and trained_models
DATA_ROOT = BULK_ROOT

# Trained model dir - where to save models
TRAINED_MODEL_DIR = BULK_ROOT + 'trained_models'

# Where to find the dataset
DB_FILE = 'path/to/subsets/folder/'

# Plaintexts and keys files
PTEXTS = np.load(DB_FILE+"test_meta.npy", mmap_mode='r')[:,0]
KEY = np.load(DB_FILE+"test_meta.npy", mmap_mode='r')[:,1]

TRACES_PER_ATTACK = 80 # Maximum number of traces to use per attack
NUM_ATTACKS = 256 # Number of attacks to average the GE over
