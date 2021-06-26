DEBUG = True

# You could get it the length of a segment, 1 s by default 
COL_SIZE = 30

# Frequency for saving checkpoint
SAVE_CHECKPOINT_FREQUENCY = 30

 # Total number of epochs
NUM_EPOCH = 100

 # Directory of data
DATA_DIR = 'wav'

 #Dir of checkpoint
CHECKPOINT_DIR = 'checkpoint'

#'Dir of logs
LOG = 'logs' 

#Batch size
BATCH_SIZE = 16

# Num steps per epoch
STEPS_PER_EPOCH = 128

#Dir of checkpoint if u use a pretrained model, None by default
LOAD_CHECKPOINT_DIR = None#'pretrained/model.149.h5'

#Initial learning rate
LR = 0.05

# Not gonna use it
SILENCE_THRESHOLD = .01 