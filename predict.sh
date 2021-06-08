python3 predict.py --SAVE_CHECKPOINT_FREQUENCY 50 \
                 --CHECKPOINT_DIR checkpoint \
                 --CATEGORIES_DIR wav/labels/labels.json \
                 --DATA_DIR eva \
                 --NUM_EPOCH 1000000 \
                 --STEPS_PER_EPOCH 1000 \
                 --COL_SIZE 45 \
                 --LOG log1 \
                 --LR 0.05 \
                 --LOAD_CHECKPOINT_DIR checkpoint/model.149.h5