DEBUG = True

# You could get it the length of a segment, 1 s by default 
COL_SIZE = 20000

OVERLAP_SIZE = 7000
# Frequency for saving checkpoint
SAVE_CHECKPOINT_FREQUENCY = 2

 # Total number of epochs
NUM_EPOCH = 100

 # Directory of data
DATA_DIR = 'test'

# Directory of evaluating data
EVALUATE_DIR = 'test'

 #Dir of checkpoint
CHECKPOINT_DIR = 'checkpoint2/model.06.h5'

#'Dir of logs
LOG = 'logs' 

#Batch size
BATCH_SIZE = 16

# Num steps per epoch
STEPS_PER_EPOCH = 128

#Dir of checkpoint if u use a pretrained model, None by default
LOAD_CHECKPOINT_DIR = None

# model dir
LOAD_MODEL_DIR = 'model'

#Initial learning rate
LR = 0.05

# Not gonna use it
SILENCE_THRESHOLD = .01 