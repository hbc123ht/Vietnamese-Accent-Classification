DEBUG = True

# You could get it the length of a segment, 1.5 s by default 
COL_SIZE = 45 

# Frequency for saving checkpoint
SAVE_CHECKPOINT_FREQUENCY = 50 

 # Total number of epochs
NUM_EPOCH = 100

 # Directory of data
DATA_DIR = 'wa'

 #Dir of checkpoint
CHECKPOINT_DIR = 'checkpoint'

#'Dir of logs
LOG = 'logs' 

#Batch size
BATCH_SIZE = 32

# Num steps per epoch
STEPS_PER_EPOCH = 128

#Dir of checkpoint if u use a pretrained model, None by default
LOAD_CHECKPOINT_DIR = 'checkpoint/model.149.h5'

#Initial learning rate
LR = 0.05

# Not gonna use it
SILENCE_THRESHOLD = .01 